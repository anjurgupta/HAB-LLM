# Harnessing Vision-Language Models for Improved Detection and Analysis of Harmful Algal Blooms (HAB)

<img src="./img/abstract.jpg" alt="Project Banner" width="800">

This repository contains data for the VLM-HAB dataset. All images are sized: 1000x1000.

Paper link: Coming soon.




## Replicating Results

### Using Replicate

To reproduce our results using Replicate:

1. Sign up for a Replicate account at [https://replicate.com](https://replicate.com)
2. Install the Replicate Python client:
   ```
   pip install replicate
   ```
3. Set up your API token:
   ```
   export REPLICATE_API_TOKEN=your_api_token_here
   ```
4. Run the following Python code:

   ```python
   import replicate

   # Example: Running a vision model
   output = replicate.run(
       "username/model-name:version",
       input={"image": open("path/to/image.jpg", "rb")}
   )
   print(output)
   ```

   Replace `"username/model-name:version"` with the specific model you want to use.

### Using Google Gemini

To replicate our results using Google Gemini:

1. Set up a Google Cloud account and enable the Gemini API
2. Install the Google Cloud SDK and authenticate
3. Install the Gemini Python client:
   ```
   pip install google-cloud-aiplatform
   ```
4. Use the following Python code:

   ```python
   from google.cloud import aiplatform

   # Initialize the Gemini cli