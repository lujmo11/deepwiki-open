"""OpenAI ModelClient integration."""

import os
import base64
from typing import (
    Dict,
    Sequence,
    Optional,
    List,
    Any,
    TypeVar,
    Callable,
    Generator,
    Union,
    Literal,
)
import re

import logging
import backoff

# optional import
from adalflow.utils.lazy_import import safe_import, OptionalPackages
from openai.types.chat.chat_completion import Choice

openai = safe_import(OptionalPackages.OPENAI.value[0], OptionalPackages.OPENAI.value[1])

from openai import OpenAI, AsyncOpenAI, Stream
from openai import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
    BadRequestError,
)
from openai.types import (
    Completion,
    CreateEmbeddingResponse,
    Image,
)
from openai.types.chat import ChatCompletionChunk, ChatCompletion, ChatCompletionMessage

from adalflow.core.model_client import ModelClient
from adalflow.core.types import (
    ModelType,
    EmbedderOutput,
    TokenLogProb,
    CompletionUsage,
    GeneratorOutput,
)
from adalflow.components.model_client.utils import parse_embedding_response

log = logging.getLogger(__name__)
T = TypeVar("T")


# ---------- helpers for parsing completions ----------
def get_first_message_content(completion: ChatCompletion) -> str:
    """Return content of the first message (default parser)."""
    log.debug(f"raw completion: {completion}")
    return completion.choices[0].message.content


def estimate_token_count(text: str) -> int:
    """Very rough token estimator based on whitespace split."""
    return len(text.split())


def parse_stream_response(completion: ChatCompletionChunk) -> str:
    """Extract delta.content from a streaming chunk."""
    return completion.choices[0].delta.content


def handle_streaming_response(generator: Stream[ChatCompletionChunk]):
    """Yield parsed content for each streaming chunk."""
    for completion in generator:
        log.debug(f"Raw chunk completion: {completion}")
        yield parse_stream_response(completion)


def get_all_messages_content(completion: ChatCompletion) -> List[str]:
    return [c.message.content for c in completion.choices]


def get_probabilities(completion: ChatCompletion) -> List[List[TokenLogProb]]:
    log_probs: List[List[TokenLogProb]] = []
    for c in completion.choices:
        content = c.logprobs.content
        log_probs_for_choice: List[TokenLogProb] = []
        for openai_token_logprob in content:
            token = openai_token_logprob.token
            logprob = openai_token_logprob.logprob
            log_probs_for_choice.append(TokenLogProb(token=token, logprob=logprob))
        log_probs.append(log_probs_for_choice)
    return log_probs


# ---------------------------- MAIN CLIENT ---------------------------- #
class OpenAIClient(ModelClient):
    """
    Component wrapper for OpenAI or Azure OpenAI.

    KISS principles:
      • Detect once whether the endpoint is Azure (`self.using_azure`)
      • Keep URL-building / logging in one helper (`_build_azure_url`)
      • Re-use the OpenAI SDK clients for actual network calls
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_completion_parser: Callable[[Completion], Any] = None,
        input_type: Literal["text", "messages"] = "text",
        base_url: Optional[str] = None,
        env_base_url_name: str = "OPENAI_API_BASE",
        env_api_key_name: str = "OPENAI_API_KEY",
    ):
        super().__init__()
        self._api_key = api_key
        self._env_api_key_name = env_api_key_name
        self._env_base_url_name = env_base_url_name

        # ---------- Endpoint & Azure flags ----------
        self.base_url = base_url or os.getenv(
            self._env_base_url_name, "https://api.openai.com/v1"
        )
        self.using_azure: bool = "azure.com" in self.base_url.lower()
        self.api_version: str = os.getenv("OPENAI_API_VERSION", "2025-01-01-preview")

        # ---------- Clients ----------
        self.sync_client = self.init_sync_client()
        self.async_client = None  # lazy-init on first async call

        # ---------- other instance fields ----------
        self.chat_completion_parser = chat_completion_parser or get_first_message_content
        self._input_type = input_type
        self._api_kwargs: Dict[str, Any] = {}

    # ------------------------ internal helpers ------------------------ #
    def _build_azure_url(self, model_type: ModelType, api_kwargs: Dict[str, Any]) -> str:
        """
        Compose the full Azure REST URL (for logging / debugging only).

        Real network calls are still executed by the AzureOpenAI SDK.
        """
        if not self.using_azure:
            return ""
        deployment = api_kwargs.get("model", "")
        url = f"{self.base_url}/openai/deployments/{deployment}"

        if model_type == ModelType.EMBEDDER:
            url += f"/embeddings?api-version={self.api_version}"
        elif model_type == ModelType.LLM:
            url += f"/chat/completions?api-version={self.api_version}"
        elif model_type == ModelType.IMAGE_GENERATION:
            url += f"/images/generations?api-version={self.api_version}"
        return url

    # ---------------------- client initialisation --------------------- #
    def init_sync_client(self):
        api_key = self._api_key or os.getenv(self._env_api_key_name)
        if not api_key:
            raise ValueError(
                f"Environment variable {self._env_api_key_name} must be set"
            )

        if self.using_azure:
            from openai import AzureOpenAI

            return AzureOpenAI(
                api_key=api_key,
                azure_endpoint=self.base_url,
                api_version=self.api_version,
            )

        return OpenAI(api_key=api_key, base_url=self.base_url)

    def init_async_client(self):
        api_key = self._api_key or os.getenv(self._env_api_key_name)
        if not api_key:
            raise ValueError(
                f"Environment variable {self._env_api_key_name} must be set"
            )

        if self.using_azure:
            from openai import AsyncAzureOpenAI

            return AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=self.base_url,
                api_version=self.api_version,
            )

        return AsyncOpenAI(api_key=api_key, base_url=self.base_url)

    # --------------------- generic parsing helpers -------------------- #
    def parse_chat_completion(
        self,
        completion: Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]],
    ) -> "GeneratorOutput":
        """Parse completion, attaching usage if available."""
        try:
            data = self.chat_completion_parser(completion)
        except Exception as e:
            log.error(f"Error parsing the completion: {e}")
            return GeneratorOutput(data=None, error=str(e), raw_response=completion)

        try:
            usage = self.track_completion_usage(completion)
            return GeneratorOutput(data=None, error=None, raw_response=data, usage=usage)
        except Exception as e:
            log.error(f"Error tracking the completion usage: {e}")
            return GeneratorOutput(data=None, error=str(e), raw_response=data)

    def track_completion_usage(
        self,
        completion: Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]],
    ) -> CompletionUsage:
        try:
            return CompletionUsage(
                completion_tokens=completion.usage.completion_tokens,
                prompt_tokens=completion.usage.prompt_tokens,
                total_tokens=completion.usage.total_tokens,
            )
        except Exception as e:
            log.error(f"Error tracking the completion usage: {e}")
            return CompletionUsage(
                completion_tokens=None, prompt_tokens=None, total_tokens=None
            )

    def parse_embedding_response(
        self, response: CreateEmbeddingResponse
    ) -> EmbedderOutput:
        try:
            return parse_embedding_response(response)
        except Exception as e:
            log.error(f"Error parsing the embedding response: {e}")
            return EmbedderOutput(data=[], error=str(e), raw_response=response)

    # -------------------- input conversion helpers -------------------- #
    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        """
        Convert component-level arguments to SDK-level kwargs.
        (Unchanged from original implementation, but left intact.)
        """
        final_model_kwargs = model_kwargs.copy()
        if model_type == ModelType.EMBEDDER:
            if isinstance(input, str):
                input = [input]
            if not isinstance(input, Sequence):
                raise TypeError("input must be a sequence of text")
            final_model_kwargs["input"] = input

        elif model_type == ModelType.LLM:
            messages: List[Dict[str, str]] = []
            images = final_model_kwargs.pop("images", None)
            detail = final_model_kwargs.pop("detail", "auto")

            if self._input_type == "messages":
                system_start_tag = "<START_OF_SYSTEM_PROMPT>"
                system_end_tag = "<END_OF_SYSTEM_PROMPT>"
                user_start_tag = "<START_OF_USER_PROMPT>"
                user_end_tag = "<END_OF_USER_PROMPT>"
                pattern = (
                    rf"{system_start_tag}\s*(.*?)\s*{system_end_tag}\s*"
                    rf"{user_start_tag}\s*(.*?)\s*{user_end_tag}"
                )
                regex = re.compile(pattern, re.DOTALL)
                match = regex.match(input) if isinstance(input, str) else None
                system_prompt, input_str = (match.group(1), match.group(2)) if match else (None, None)

                if system_prompt and input_str:
                    messages.append({"role": "system", "content": system_prompt})
                    if images:
                        content = [{"type": "text", "text": input_str}]
                        if isinstance(images, (str, dict)):
                            images = [images]
                        for img in images:
                            content.append(self._prepare_image_content(img, detail))
                        messages.append({"role": "user", "content": content})
                    else:
                        messages.append({"role": "user", "content": input_str})

            if len(messages) == 0:
                if images:
                    content = [{"type": "text", "text": input}]
                    if isinstance(images, (str, dict)):
                        images = [images]
                    for img in images:
                        content.append(self._prepare_image_content(img, detail))
                    messages.append({"role": "user", "content": content})
                else:
                    messages.append({"role": "user", "content": input})
            final_model_kwargs["messages"] = messages

        elif model_type == ModelType.IMAGE_GENERATION:
            final_model_kwargs["prompt"] = input
            if "model" not in final_model_kwargs:
                raise ValueError("model must be specified for image generation")
            final_model_kwargs["size"] = final_model_kwargs.get("size", "1024x1024")
            final_model_kwargs["quality"] = final_model_kwargs.get("quality", "standard")
            final_model_kwargs["n"] = final_model_kwargs.get("n", 1)
            final_model_kwargs["response_format"] = final_model_kwargs.get("response_format", "url")

            image = final_model_kwargs.get("image")
            if isinstance(image, str) and os.path.isfile(image):
                final_model_kwargs["image"] = self._encode_image(image)
            mask = final_model_kwargs.get("mask")
            if isinstance(mask, str) and os.path.isfile(mask):
                final_model_kwargs["mask"] = self._encode_image(mask)
        else:
            raise ValueError(f"model_type {model_type} is not supported")

        return final_model_kwargs

    # ---------------------- main sync / async calls ---------------------- #
    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
            BadRequestError,
        ),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        full_url = self._build_azure_url(model_type, api_kwargs)
        if full_url:
            log.debug(f"Azure OpenAI full URL: {full_url}")

        self._api_kwargs = api_kwargs
        if model_type == ModelType.EMBEDDER:
            return self.sync_client.embeddings.create(**api_kwargs)

        elif model_type == ModelType.LLM:
            if api_kwargs.get("stream"):
                log.debug("streaming call")
                self.chat_completion_parser = handle_streaming_response
                return self.sync_client.chat.completions.create(**api_kwargs)

            # non-streaming converted to streaming under the hood
            log.debug("non-streaming call converted to streaming")
            streaming_kwargs = api_kwargs.copy()
            streaming_kwargs["stream"] = True
            stream_response = self.sync_client.chat.completions.create(**streaming_kwargs)

            accumulated_content = ""
            _id, _model, _created = "", "", 0
            for chunk in stream_response:
                _id = getattr(chunk, "id", _id)
                _model = getattr(chunk, "model", _model)
                _created = getattr(chunk, "created", _created)
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    accumulated_content += delta.content

            return ChatCompletion(
                id=_id,
                model=_model,
                created=_created,
                object="chat.completion",
                choices=[
                    Choice(
                        index=0,
                        finish_reason="stop",
                        message=ChatCompletionMessage(content=accumulated_content, role="assistant"),
                    )
                ],
            )

        elif model_type == ModelType.IMAGE_GENERATION:
            if "image" in api_kwargs:
                if "mask" in api_kwargs:
                    response = self.sync_client.images.edit(**api_kwargs)
                else:
                    response = self.sync_client.images.create_variation(**api_kwargs)
            else:
                response = self.sync_client.images.generate(**api_kwargs)
            return response.data

        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
            BadRequestError,
        ),
        max_time=5,
    )
    async def acall(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        full_url = self._build_azure_url(model_type, api_kwargs)
        if full_url:
            log.debug(f"Azure OpenAI full URL: {full_url}")

        self._api_kwargs = api_kwargs
        if self.async_client is None:
            self.async_client = self.init_async_client()

        if model_type == ModelType.EMBEDDER:
            return await self.async_client.embeddings.create(**api_kwargs)

        elif model_type == ModelType.LLM:
            return await self.async_client.chat.completions.create(**api_kwargs)

        elif model_type == ModelType.IMAGE_GENERATION:
            if "image" in api_kwargs:
                if "mask" in api_kwargs:
                    response = await self.async_client.images.edit(**api_kwargs)
                else:
                    response = await self.async_client.images.create_variation(**api_kwargs)
            else:
                response = await self.async_client.images.generate(**api_kwargs)
            return response.data

        else:
            raise ValueError(f"model_type {model_type} is not supported")

    # ------------------------- serialisation ------------------------- #
    @classmethod
    def from_dict(cls: type[T], data: Dict[str, Any]) -> T:
        obj = super().from_dict(data)
        obj.sync_client = obj.init_sync_client()
        obj.async_client = obj.init_async_client()
        return obj

    def to_dict(self) -> Dict[str, Any]:
        exclude = ["sync_client", "async_client"]
        return super().to_dict(exclude=exclude)

    # ------------------------ image utilities ------------------------ #
    def _encode_image(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError:
            raise ValueError(f"Image file not found: {image_path}")
        except PermissionError:
            raise ValueError(f"Permission denied when reading image file: {image_path}")
        except Exception as e:
            raise ValueError(f"Error encoding image {image_path}: {str(e)}")

    def _prepare_image_content(
        self, image_source: Union[str, Dict[str, Any]], detail: str = "auto"
    ) -> Dict[str, Any]:
        if isinstance(image_source, str):
            if image_source.startswith(("http://", "https://")):
                return {"type": "image_url", "image_url": {"url": image_source, "detail": detail}}
            else:
                base64_image = self._encode_image(image_source)
                return {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": detail},
                }
        return image_source


# --------------------------- simple demo --------------------------- #
if __name__ == "__main__":
    from adalflow.core import Generator
    from adalflow.utils import setup_env

    setup_env()
    gen = Generator(
        model_client=OpenAIClient(),
        model_kwargs={"model": "gpt-4o", "stream": False},
    )
    print(gen({"input_str": "What is the meaning of life?"}))