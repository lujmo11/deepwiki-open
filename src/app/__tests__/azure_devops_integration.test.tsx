// @jest-environment jsdom
import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import Page from "../page";
import fetchMock from "jest-fetch-mock";

beforeAll(() => {
  fetchMock.enableMocks();
});
beforeEach(() => {
  fetchMock.resetMocks();
});

describe("Azure DevOps Integration", () => {
  it("displays repo structure for valid Azure DevOps URL", async () => {
    fetchMock.mockResponseOnce(
      JSON.stringify({ tree: [{ path: "/README.md" }] })
    );
    render(<Page />);
    const input = screen.getByPlaceholderText(/repository url/i);
    fireEvent.change(input, {
      target: { value: "https://dev.azure.com/org/proj/_git/repo" },
    });
    fireEvent.click(screen.getByText(/load/i));
    await waitFor(() =>
      expect(screen.getByText("/README.md")).toBeInTheDocument()
    );
  });

  it("shows error for invalid PAT", async () => {
    fetchMock.mockRejectOnce(new Error("Authentication failed"));
    render(<Page />);
    const input = screen.getByPlaceholderText(/repository url/i);
    fireEvent.change(input, {
      target: { value: "https://dev.azure.com/org/proj/_git/repo" },
    });
    fireEvent.click(screen.getByText(/load/i));
    await waitFor(() =>
      expect(screen.getByText(/authentication failed/i)).toBeInTheDocument()
    );
  });

  it("shows error for repo not found", async () => {
    fetchMock.mockResponseOnce(
      JSON.stringify({ detail: "Resource not found" }),
      { status: 404 }
    );
    render(<Page />);
    const input = screen.getByPlaceholderText(/repository url/i);
    fireEvent.change(input, {
      target: { value: "https://dev.azure.com/org/proj/_git/nonexistent" },
    });
    fireEvent.click(screen.getByText(/load/i));
    await waitFor(() =>
      expect(screen.getByText(/not found/i)).toBeInTheDocument()
    );
  });
});