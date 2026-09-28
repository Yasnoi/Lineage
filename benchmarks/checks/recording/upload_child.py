"""Trusted child used only by the upload shutdown contract."""


from benchmarks.checks.support.tracing import hanging_upload_child
if __name__ == "__main__":
    hanging_upload_child()
