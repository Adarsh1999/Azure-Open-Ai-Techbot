# GPT 4o Tech Bot 🤖

### Modified from https://github.com/jainal09/Azure-Open-Ai-Techbot


<table>
  <tr>
    <td>
      <img src="https://github.com/jainal09/Azure-Open-Ai-Techbot/assets/34179361/0b2f194e-f0e8-4c35-bb60-40859a73f3ad" alt="tech-bot" width="600">
    </td>
    <td>
      This is a tech bot that efficiently answers questions related to technology. It is programmed to specialize in answering technological questions, simulating specific use cases of a particular topic. The bot utilizes the GPT-4o API and streamlit for its functionality.
    </td>
  </tr>
</table>


https://github.com/jainal09/Azure-Open-Ai-Techbot/assets/34179361/d42bd484-48e0-41bb-8eec-3629dddf595c

## Built With 🛠️
- [GPT-4o](https://openai.com/index/hello-gpt-4o)
- [Streamlit](https://streamlit.io/)
- [Azure Open AI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)

## Getting Started 🚀

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

1. Python 3.11 or higher
2. Poetry


### Installing

Assuming you have python and poetry installed, you can install the dependencies by running the following command:

```bash
poetry install
```

### Setting up the environment
#### Azure Open AI
1. Create an account on [Azure Open AI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
2. Create a new project and grab the API key
   
#### Environment variables
1. Create a `.env` file in the root directory and add the following line:
```bash
API_KEY=YOUR_API_KEY
AZURE_ENDPOINT=YOUR_AZURE_ENDPOINT
API_VERSION=2024-02-01
AZURE_DEPLOYMENT=YOUR_AZURE_DEPLOYMENT
```


### Running the bot
```bash
python -m streamlit run src/tech_bot/main.py
```
This will start a server at `http://localhost:8501` where you can interact with the bot.

## Features 📋
1. Real time chat GPT like interface with the bot 💬
2. Streaming response from the GPT-4o API to the user 🌐
3. Safeguards for inappropriate content 🚫
4. Features to restrict Answers to questions only for the domain of technology achieved through prompt engineering ⚙️
5. Real time token calculation for displaying the token usage to the user 📊
6. Feature to pass the context and history to the GPT-4o API for better responses 📚 


## Demo Video 📺
Click on the image below to watch the demo video on YouTube:

[![Youtube Video](https://img.youtube.com/vi/BB6yraGmftk/0.jpg)](https://www.youtube.com/watch?v=BB6yraGmftk)


## Contributing 🖇️

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License 📄

This project is licensed under the [LICENSE.md](LICENSE.md) - see the file for details


