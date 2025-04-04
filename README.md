# TeleBot Explainer

A Telegram bot that automatically explains and summarizes posts from a main channel when they're forwarded to discussion groups.

## Features

- Auto-explains posts when forwarded from the main channel to discussion groups
- Includes Perplexity.ai search links for additional context
- `/explain` command to get detailed explanations of any message
- `/summarise` command to get concise summaries of any message
- Works with text, images, and captions

## Requirements

- Python 3.7+
- Telegram Bot API token
- Google Gemini API key

## Installation

```bash
git clone https://github.com/yourusername/telebot-explainer.git
cd telebot-explainer
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with the following variables:

```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
GEMINI_API_KEY=your_gemini_api_key
MAIN_CHANNEL_ID=your_channel_id
DISCUSSION_GROUP_IDS=group_id1,group_id2,group_id3
```

## Usage

1. Add the bot to your main channel and all discussion groups
2. Ensure the bot has permission to read messages
3. Run the bot: `python bot.py`
4. Posts from the main channel will be automatically explained when forwarded to discussion groups
5. Users can reply to any message with `/explain` or `/summarise` in discussion groups

## Note

For the bot to see all messages in groups, disable Privacy Mode when creating your bot with @BotFather. 