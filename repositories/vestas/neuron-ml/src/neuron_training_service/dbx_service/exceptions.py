class DBXJobDoesNotExistError(Exception):
    def __init__(self, job_run_id: int):
        self.job_run_id = job_run_id
        super().__init__(f"Job run with ID {job_run_id} does not exist.")
