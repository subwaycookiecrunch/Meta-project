from openenv.core.env_server import create_fastapi_app
from code_review_env.server.environment import CodeReviewEnvironment
from code_review_env.models import CodeReviewAction, CodeReviewObservation

app = create_fastapi_app(CodeReviewEnvironment, CodeReviewAction, CodeReviewObservation)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
