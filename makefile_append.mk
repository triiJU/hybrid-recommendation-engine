api:
	uvicorn api.app:app --reload --port 8080

docker-build:
	docker build -t hybrid-recommender:latest .

docker-run:
	docker run -p 8080:8080 hybrid-recommender:latest
