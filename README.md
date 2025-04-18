Instructions to run the app
- docker pull omprakash/dance-comparison
- mkdir dance_videos       [ put the benchmark video here]
- docker run -p 5001:5001 \
  -v $(pwd)/dance_videos:/app/dance_videos \
  --privileged \
  omprakash912/dance-comparison
