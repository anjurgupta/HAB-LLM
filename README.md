# VLM-HAB: Dataset or Vision Language Models

<img src="./images/header.tif" alt="Project Banner" width="800" height="400">

![Project Banner](./images/header.jpg)

## Summary

This repository contains data for VLM-HAB dataset. all The images are size:  1000*1000

Key features:
- [Feature 1]
- [Feature 2]
- [Feature 3]




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