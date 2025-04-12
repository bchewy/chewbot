import logging
import os
import asyncio
import google.generativeai as genai
from urllib.parse import quote_plus
from dotenv import load_dotenv
from telegram import Update, ReplyParameters, PhotoSize, File
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, filters as Filters # Alias filters for clarity
from typing import List, Optional
import re
import json

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#TARGET_CHAT_IDS_STR = os.getenv("TARGET_CHAT_ID") # Old variable
MAIN_CHANNEL_ID = os.getenv("MAIN_CHANNEL_ID")
DISCUSSION_GROUP_IDS_STR = os.getenv("DISCUSSION_GROUP_IDS")

# Configure logging
# Define log file path
LOG_FILE = "bot.log"

# Basic config for console
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Create a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Ensure logger level is set

# Prevent double logging if basicConfig already added a handler
if logger.hasHandlers():
    logger.handlers.clear()

# Console Handler (optional, if you still want console output)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)

# File Handler (JSON)
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record['exc_info'] = self.formatException(record.exc_info)
        return json.dumps(log_record)

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(JsonFormatter())
logger.addHandler(file_handler)

logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Anonymous Naming ---
CHEW_NAMES = [
    "Chewer 🦷", "Chewy 🍬", "Chewbert 🤖", "Chewbacca 🐻", "Chewzilla 🦖",
    "Chewster 😎", "Chewliette 💃", "SirChewsAlot 🏇", "Chewmeister 🧙‍♂️",
    "Chewbacca Jr. 🐾", "MegaChew 💥", "Captain Chew 🧢", "Agent Chew 🕶️",
    "Professor Chew 👓", "Doctor Chew 🩺", "ChewBro 🧍‍♂️", "ChewchewTrain 🚂",
    "Chewrito 🌯", "ChewManji 🎲", "Chewlexa 📢", "Chewbiscuits 🍪", "Chewrambo 🎖️",
    "Chewtopher ✨", "Chewbroski 🧊", "Chewbeard 🧔", "Chewthulu 🐙", "Chewnicorn 🦄",
    "Chewpocalypse 🔥", "Chewfinity ♾️", "ChewMysterio 🌀", "Chewlock Holmes 🕵️‍♂️",
    "Chewtron ⚡", "Chewzilla Returns 🎬", "Chewraffe 🦒", "Chewmander 🐉",
    "Chewshroom 🍄", "Chewbubbles 🫧", "Chewkoala 🐨", "Chewdozer 🚜",
    "Chewgoose 🪿", "Chewtato 🥔", "Chewstorm 🌪️", "Chewbug 🐛", "Chewnado 🌪️",
    "Chewtoise 🐢", "Chewperman 🦸‍♂️", "Chewzilla X 🚀", "Chewmonaut 👨‍🚀",
    "Chewbear 🧸", "Chewgeneer 🧑‍🔧", "Chewluminati 👁️", "Chewbean 🫘", "Chewtopia 🏝️",
    "Chewzilla Jr. 👶🦖", "Chewblade 🗡️", "Chewberine 🐺", "Chewjam 🏀", "Chewboss 👔",
    "Chewcraft 🧱", "Chewrrior 🛡️", "Chewzen 🧘‍♂️", "Chewminator 🤖", "Chewberry 🍓",
    "Chewbathor ⚡🔨", "Chewphant 🐘", "Chewverse 🌌", "Chewpie 🥧", "Chewspresso ☕",
    "Chewstache 👨‍🦰", "Chewlion 🦁", "Chewrocket 🚀", "Chewbeam 🔦", "Chewpop 🍭",
    "Chewspike 🌵", "Chewchamp 🏆", "Chewbeans 🌱", "Chewphantom 👻", "Chewtronics 💻",
    "Chewzilla Prime 🔱", "Chewbrawl 🥊", "Chewdini 🎩", "Chewlord 🧛", "Chewtastic 🌟",
    "Chewzilla 9000 🤯", "Chewcrash 💥", "Chewclops 👁️", "Chewfinity Gauntlet 🧤",
    "Chewhawk 🦅", "Chewblade Runner 🏃‍♂️", "Chewtank 🛡️", "Chewquake 🌍", "Chewmaster 🧠",
    "ChewBaron 💼", "ChewKnight ⚔️", "Chewboy 🤠", "Chewsicle 🍡", "Chewzanator 🔧",
    "Chewrage 😤", "Chewlebrity 📸", "Chewception 🌀", "Chewpocalypse Now ☢️",
    "Chewblitz ⚡", "Chewnami 🌊", "Chewbot 🤖", "Chewborg 🦾", "Chewzilla Max 🛸"
]

# Process Discussion Group IDs
DISCUSSION_GROUP_IDS: List[str] = []
if DISCUSSION_GROUP_IDS_STR:
    DISCUSSION_GROUP_IDS = [chat_id.strip() for chat_id in DISCUSSION_GROUP_IDS_STR.split(',') if chat_id.strip()]
    logger.info(f"Allowed Discussion Group IDs: {DISCUSSION_GROUP_IDS}")
else:
    logger.warning("DISCUSSION_GROUP_IDS environment variable not set or empty.")

# Validate Main Channel ID
if not MAIN_CHANNEL_ID:
    logger.warning("MAIN_CHANNEL_ID environment variable not set or empty.")
    # Decide if this is critical - bot might only do manual commands if no channel ID

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash') # Or use 'gemini-pro'
else:
    logger.error("GEMINI_API_KEY environment variable not set.")
    model = None # Handle cases where Gemini is not configured

# --- Prompt Templates ---
EXPLAIN_PROMPT_TEMPLATE = """
You are an AI assistant helping users understand posts in a Telegram channel about AI/cloud/tech news, random GitHub repos, and niche tech topics.
The user is likely an expert but may not be familiar with the specific topic mentioned.
Explain the following post content concisely, assuming an expert audience. Focus on the core concepts, potential impact, or why it might be interesting/noteworthy.
If it's a link (especially GitHub), briefly describe the project and its purpose.
Keep the explanation brief and informative.

Post Content:
---
{message_text}
---

Explanation:
"""

# Renamed and updated prompt template
AUTO_EXPLAIN_PROMPT_TEMPLATE = """
You are an AI assistant helping users understand posts in a Telegram channel about AI/cloud/tech news, random GitHub repos, and niche tech topics.
Explain the core concepts or technology mentioned in the following post content using clear language, possibly with bullet points for key aspects. Assume an expert audience but clarify potentially niche terms or acronyms concisely.
Focus on *what it is* and *why it's noteworthy*. 
**Do not use any Markdown formatting in your response.**

Post Content:
---
{message_text}
---

Explanation:
"""

# --- New Prompt Template for Perplexity Query ---
PERPLEXITY_QUERY_PROMPT_TEMPLATE = """
Based on the following original post content and its explanation, generate a concise and effective search query (max 10 words) suitable for Perplexity.ai to find more information about the main topic or technology discussed. Focus on keywords and core concepts. Output only the search query itself, nothing else.

Original Post Content:
---
{original_content}
---

Explanation:
---
{explanation}
---

Perplexity Search Query:
"""

SUMMARISE_PROMPT_TEMPLATE = """
You are an AI assistant helping users quickly grasp the essence of posts in a Telegram channel about AI/cloud/tech news, random GitHub repos, and niche tech topics.
Summarize the following post content into a single, concise paragraph (max 2-3 sentences). Focus on the main takeaway or key information.

Post Content:
---
{message_text}
---

Summary:
"""

# --- Helper Function ---

def escape_markdown_v2(text: str) -> str:
    """Escapes characters for Telegram MarkdownV2.
    
    See: https://core.telegram.org/bots/api#markdownv2-style
    """
    # Characters to escape
    escape_chars = r'_ * [ ] ( ) ~ ` > # + - = | { } . !'
    # Use a regex substitution to backslash-escape them
    # Corrected replacement using backreference \1
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

async def _process_command(update: Update, context: ContextTypes.DEFAULT_TYPE, action: str, prompt_template: str):
    """Generic handler for reply-based commands like /explain and /summarise. Anonymizes the command."""
    command = f"/{action}"
    command_message = update.message
    chat_id = command_message.chat_id
    user_id = command_message.from_user.id
    command_message_id = command_message.message_id
    bot_id = context.bot.id

    # Should not happen for user commands, but safeguard
    if user_id == bot_id:
        return

    logger.info(f"Received {command} command (msg_id: {command_message_id}) from user {user_id} in chat {chat_id}")

    if not model:
        logger.error(f"Gemini model not configured. Cannot process {command}.")
        # Try to delete the command even if model fails
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=command_message_id)
            logger.info(f"Deleted command message {command_message_id} even though model is not configured.")
        except Exception as delete_err:
            logger.error(f"Failed to delete command message {command_message_id} after model config check: {delete_err}")
        await update.message.reply_text("Sorry, the AI backend is not configured.")
        return

    # Check if the command is a reply
    original_message = command_message.reply_to_message
    if not original_message:
        # Don't delete if it wasn't a valid reply command initially
        await command_message.reply_text(f"Please reply to the message you want me to {action} with {command}.")
        return

    # Check if the command is in an allowed discussion group
    if str(chat_id) not in DISCUSSION_GROUP_IDS:
        logger.warning(f"Ignoring {command} in non-discussion group chat: {chat_id}")
        # Don't delete if it's not in the right group
        return

    # --- Anonymize the command message ---
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=command_message_id)
        logger.info(f"Successfully deleted command message {command_message_id}")
    except Exception as delete_err:
        logger.error(f"Failed to delete command message {command_message_id}: {delete_err}", exc_info=True)
        # Proceed even if deletion fails, but log it.

    # --- Process the original message ---
    message_text = original_message.text or original_message.caption # Handle text and captions
    original_message_id = original_message.message_id # ID of the message being explained/summarised

    if not message_text:
        logger.warning(f"{command} used on message {original_message_id} with no text/caption.")
        # Send reply to the original message's context
        await context.bot.send_message(chat_id=chat_id,
                                       text=f"Sorry, I can only {action} messages with text content or captions.",
                                       reply_parameters=ReplyParameters(message_id=original_message_id))
        return

    logger.info(f"Attempting to {action} original message ID {original_message_id}")
    # Indicate processing (typing action in the chat)
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    except Exception as e:
        logger.warning(f"Could not send typing action in chat {chat_id}: {e}")

    try:
        # Prepare the prompt for Gemini using the original message's text
        prompt = prompt_template.format(message_text=message_text)

        response = await model.generate_content_async(prompt)
        result_text = response.text

        logger.info(f"Generated {action} for original message ID {original_message_id}")

        # Reply to the ORIGINAL message the user replied to, not the deleted command
        await context.bot.send_message(
            chat_id=chat_id,
            text=result_text,
            reply_parameters=ReplyParameters(message_id=original_message_id) # Reply to original post
        )

    except Exception as e:
        logger.error(f"Error generating {action} for original message ID {original_message_id}: {e}", exc_info=True)
        try:
            # Try to send error message back to the chat, replying to the original message
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Sorry, I encountered an error trying to {action} that: {type(e).__name__}",
                reply_parameters=ReplyParameters(message_id=original_message_id) # Reply to original post
            )
        except Exception as send_error:
            logger.error(f"Failed to send error message to chat {chat_id}: {send_error}", exc_info=True)


# --- Auto Handler ---
async def auto_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Automatically summarizes new channel posts (text and/or image) in linked discussion groups."""
    logger.info("auto_summary triggered.")

    if not update.channel_post:
        logger.warning("auto_summary: Received update without channel_post.")
        return

    channel_post = update.channel_post

    # Check if the post is from the designated MAIN_CHANNEL_ID
    if not MAIN_CHANNEL_ID or str(channel_post.chat.id) != MAIN_CHANNEL_ID:
        logger.info(f"auto_summary: Ignoring post from non-main channel: {channel_post.chat.id}. Expected: {MAIN_CHANNEL_ID}")
        return

    logger.info(f"auto_summary: Processing post ID {channel_post.message_id} from main channel {channel_post.chat.id} ({channel_post.chat.title or 'No Title'}).")

    photo_caption = channel_post.text or channel_post.caption
    has_photo = bool(channel_post.photo)

    if not photo_caption and not has_photo:
        logger.info("auto_summary: Post has no text, caption, or photo. Ignoring.")
        return

    if not model:
        logger.warning("auto_summary: Gemini model not configured. Skipping.")
        return

    if not DISCUSSION_GROUP_IDS:
        logger.warning("auto_summary: No discussion group IDs configured. Skipping.")
        return

    logger.info(f"auto_summary: Attempting summary generation for post {channel_post.message_id} (Has Photo: {has_photo}, Has Caption: {bool(photo_caption)}).")

    try:
        # --- Prepare Gemini Input ---
        # Use caption for text context, add placeholder if only image exists
        prompt_context_text = photo_caption if photo_caption else "[Image Content Only]"
        # Use the correct template for the explanation prompt
        prompt_for_gemini = AUTO_EXPLAIN_PROMPT_TEMPLATE.format(message_text=prompt_context_text)

        gemini_payload = [prompt_for_gemini] # Start with the overall instruction/prompt

        image_bytes = None
        if has_photo:
            try:
                # Get the largest photo
                photo_size: PhotoSize = channel_post.photo[-1]
                photo_file: File = await context.bot.get_file(photo_size.file_id)
                image_bytes_bytearray = await photo_file.download_as_bytearray()
                image_bytes = bytes(image_bytes_bytearray) # Convert to standard bytes
                logger.info(f"auto_summary: Downloaded photo {photo_file.file_id} ({len(image_bytes)} bytes)")

                # Create Blob for Gemini API (assuming JPEG for simplicity)
                # TODO: Ideally detect mime type if possible/needed
                image_blob = {"mime_type": "image/jpeg", "data": image_bytes}
                gemini_payload.append(image_blob) # Add image after the prompt
                logger.info("auto_summary: Added image to Gemini payload.")

            except Exception as img_err:
                logger.error(f"auto_summary: Failed to download/process photo for post {channel_post.message_id}: {img_err}", exc_info=True)
                # Proceed with text only if image failed but caption exists
                if not photo_caption:
                    logger.warning("auto_summary: Skipping post as image failed and no caption exists.")
                    return # Skip if only image and it failed

        # --- Call Gemini ---
        if not gemini_payload[1:]: # Check if only the prompt text is present (no image added)
             if not photo_caption: # If no image was successfully added AND no caption, nothing to process
                  logger.warning("auto_summary: No processable content (text/caption or image). Skipping.")
                  return

        logger.info(f"auto_summary: Sending payload to Gemini (Prompt + {len(gemini_payload)-1} other parts).")
        # Call Gemini correctly without prompt in generation_config
        response_explain = await model.generate_content_async(gemini_payload)
        explanation_text = response_explain.text
        logger.info(f"auto_summary: Generated explanation for post {channel_post.message_id}. Length: {len(explanation_text)}")

        # --- Second Gemini Call: Generate Perplexity Query ---
        perplexity_query = None
        # Only generate query if there was some original text/caption to base it on
        if photo_caption:
            try:
                logger.info(f"auto_summary: Generating Perplexity query for post {channel_post.message_id}.")
                perplexity_prompt = PERPLEXITY_QUERY_PROMPT_TEMPLATE.format(
                    original_content=photo_caption,
                    explanation=explanation_text
                )
                response_query = await model.generate_content_async(perplexity_prompt)
                perplexity_query = response_query.text.strip() # Get query and strip whitespace
                logger.info(f"auto_summary: Generated Perplexity query: '{perplexity_query}'")
            except Exception as query_err:
                logger.error(f"auto_summary: Failed to generate Perplexity query: {query_err}", exc_info=True)
                perplexity_query = photo_caption # Fallback to original text if query generation fails
        else:
             logger.info(f"auto_summary: Skipping Perplexity query generation as there was no original text.")
             perplexity_query = "AI analysis of image content" # Generic fallback for image-only posts

        # --- Store Result --- Use new key
        if 'pending_auto_explanations' not in context.bot_data:
            context.bot_data['pending_auto_explanations'] = {}
        context.bot_data['pending_auto_explanations'][channel_post.message_id] = {
            'explanation': explanation_text,
            'perplexity_query': perplexity_query
        }
        logger.info(f"auto_summary: Stored pending explanation and perplexity query for channel post {channel_post.message_id}")

    except Exception as e:
        logger.error(f"auto_summary: Error during explanation generation for post {channel_post.message_id}: {e}", exc_info=True)


# --- Discussion Group Handler ---
async def handle_discussion_forward(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles forwarded messages in the discussion group to find the mirrored post and reply."""
    message = update.message
    logger.info(f"handle_discussion_forward: Received message {message.message_id} in group {message.chat.id}")

    # Check if it's a forward and if we have pending auto-explanations
    if not message.forward_origin or 'pending_auto_explanations' not in context.bot_data or not context.bot_data['pending_auto_explanations']:
        logger.debug("handle_discussion_forward: Not a relevant forward or no pending explanations.")
        return

    # Check if the forward is from our main channel
    if str(message.forward_origin.chat.id) != MAIN_CHANNEL_ID:
        logger.debug(f"handle_discussion_forward: Forward is not from main channel {MAIN_CHANNEL_ID}")
        return

    original_channel_post_id = message.forward_origin.message_id
    logger.info(f"handle_discussion_forward: Message {message.message_id} is a forward of channel post {original_channel_post_id}")

    # Check if we have a pending explanation for this original post ID
    if original_channel_post_id in context.bot_data['pending_auto_explanations']:
        pending_data = context.bot_data['pending_auto_explanations'].pop(original_channel_post_id)
        explanation_text = pending_data['explanation']
        # Retrieve the generated query
        perplexity_query = pending_data['perplexity_query']

        logger.info(f"handle_discussion_forward: Found pending explanation for channel post {original_channel_post_id}. Replying to discussion message {message.message_id}.")

        # Create Perplexity link using the generated query
        perplexity_link_text = ""
        if perplexity_query:
            encoded_query = quote_plus(perplexity_query)
            perplexity_url = f"https://perplexity.ai/search?q={encoded_query}"
            perplexity_link_text = f"Explore on Perplexity: {perplexity_url}\n"
        else:
            # Fallback if query generation failed or wasn't applicable
            perplexity_link_text = "(Could not generate specific Perplexity link)\n"

        # Format the final plain text message
        reply_title = "Explanation:\n"
        explanation_section = f"\n{explanation_text}" # Use raw explanation text
        summary_prompt_text = "\n\n(For a brief summary, reply /summarise)"

        full_reply_text = f"{reply_title}{perplexity_link_text}{explanation_section}{summary_prompt_text}"

        try:
            await context.bot.send_message(
                chat_id=message.chat.id,
                text=full_reply_text,
                reply_to_message_id=message.message_id,
                disable_web_page_preview=True
            )
            logger.info(f"handle_discussion_forward: Successfully sent explanation+link for channel post {original_channel_post_id} as reply to discussion msg {message.message_id}")
        except Exception as send_error:
            logger.error(f"handle_discussion_forward: Failed to send explanation+link reply to discussion msg {message.message_id}: {send_error}", exc_info=True)
            # Put the data back if sending failed
            context.bot_data['pending_auto_explanations'][original_channel_post_id] = pending_data
    else:
        logger.debug(f"handle_discussion_forward: No pending explanation found for channel post {original_channel_post_id}.")


# --- Anonymous Comment Handler ---
async def handle_anonymous_comment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles regular messages in discussion groups to make them anonymous."""
    message = update.message
    chat_id = message.chat_id
    user_id = message.from_user.id
    bot_id = context.bot.id

    # Ignore messages from the bot itself
    if user_id == bot_id:
        return

    logger.info(f"handle_anonymous_comment: Received message {message.message_id} from user {user_id} in group {chat_id}")

    # Basic check if it's in a configured discussion group (already filtered, but good practice)
    if str(chat_id) not in DISCUSSION_GROUP_IDS:
        logger.warning(f"handle_anonymous_comment: Message {message.message_id} received in non-discussion group {chat_id}. Filter should have caught this.")
        return

    original_message_id = message.message_id
    reply_to_id = message.reply_to_message.message_id if message.reply_to_message else None
    content_type = message.effective_attachment.__class__.__name__ if message.effective_attachment else "text"

    logger.info(f"handle_anonymous_comment: Anonymizing message {original_message_id} (type: {content_type}) replying to {reply_to_id}")

    # --- Assign or retrieve anonymous name ---
    if 'user_anon_names' not in context.bot_data:
        context.bot_data['user_anon_names'] = {}
    if 'anon_name_index' not in context.bot_data:
        context.bot_data['anon_name_index'] = 0

    user_anon_map = context.bot_data['user_anon_names']
    anon_name = ""

    if user_id in user_anon_map:
        anon_name = user_anon_map[user_id]
        logger.debug(f"Found existing anon name for user {user_id}: {anon_name}")
    else:
        # Assign a new name
        name_index = context.bot_data['anon_name_index']
        anon_name = CHEW_NAMES[name_index % len(CHEW_NAMES)] # Cycle through names
        user_anon_map[user_id] = anon_name
        context.bot_data['anon_name_index'] = name_index + 1 # Increment for next user
        # Mark bot_data as modified if using certain persistence backends (though default PicklePersistence usually handles this)
        context.bot_data.setdefault('_modified', True)
        logger.info(f"Assigned new anon name to user {user_id}: {anon_name} (Index: {name_index})")

    anon_prefix = f"[{anon_name}]: " # Use the assigned/retrieved name

    try:
        # 1. Delete the original message
        await context.bot.delete_message(chat_id=chat_id, message_id=original_message_id)
        logger.info(f"handle_anonymous_comment: Deleted original message {original_message_id}")

        # 2. Repost the content anonymously with prefix
        if message.text:
            await context.bot.send_message(chat_id=chat_id, text=f"{anon_prefix}{message.text}", reply_to_message_id=reply_to_id)
        elif message.photo:
            # Send the largest photo with prefixed caption
            caption_text = f"{anon_prefix}{message.caption}" if message.caption else anon_prefix.strip() # Add prefix, handle None caption
            await context.bot.send_photo(chat_id=chat_id, photo=message.photo[-1].file_id, caption=caption_text, reply_to_message_id=reply_to_id)
        elif message.sticker:
             # Stickers don't have text/captions to prefix easily. Send as is for now.
             # Optionally, could send a separate text message attributing it, but might be noisy.
             await context.bot.send_sticker(chat_id=chat_id, sticker=message.sticker.file_id, reply_to_message_id=reply_to_id)
        # Add more handlers for other types (video, voice, document etc.) if needed
        # elif message.video:
        #     caption_text = f"{anon_prefix}{message.caption}" if message.caption else anon_prefix.strip()
        #     await context.bot.send_video(chat_id=chat_id, video=message.video.file_id, caption=message.caption, reply_to_message_id=reply_to_id)
        else:
            logger.warning(f"handle_anonymous_comment: Unsupported message type '{content_type}' for anonymization in message {original_message_id}. Original deleted.")
            # Optionally send a placeholder message with prefix
            await context.bot.send_message(chat_id=chat_id, text=f"{anon_prefix}[Anonymized message of unsupported type]", reply_to_message_id=reply_to_id)

        logger.info(f"handle_anonymous_comment: Successfully anonymized message {original_message_id} with prefix '{anon_prefix.strip()}'")

    except Exception as e:
        logger.error(f"handle_anonymous_comment: Failed to anonymize message {original_message_id}: {e}", exc_info=True)
        # Decide if you want to notify the user or group about the failure
        # Example: await context.bot.send_message(chat_id=chat_id, text=f"Debug: Failed to anonymize message {original_message_id}. Error: {e}", reply_to_message_id=reply_to_id)


# --- Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on /start"""
    await update.message.reply_text("Hi! Reply to a post in an allowed comment thread with /explain or /summarise.")

async def explain_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Explains the message this command replies to."""
    await _process_command(update, context, action="explain", prompt_template=EXPLAIN_PROMPT_TEMPLATE)

async def summarise_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Summarises the message this command replies to."""
    await _process_command(update, context, action="summarise", prompt_template=SUMMARISE_PROMPT_TEMPLATE)


async def main() -> None: # Make main async
    """Start the bot."""
    # --- Pre-run Checks ---
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("TELEGRAM_BOT_TOKEN environment variable not set. Exiting.")
        return # Or raise ValueError
    if not GEMINI_API_KEY:
        logger.critical("GEMINI_API_KEY environment variable not set. AI features disabled.")
        # Decide if the bot should run without AI or exit
        # return # Uncomment to exit if Gemini is mandatory
    if not DISCUSSION_GROUP_IDS:
        # Commands require discussion groups, auto-summary might too depending on config
        logger.critical("DISCUSSION_GROUP_IDS environment variable not set or empty. Bot may not function correctly. Exiting.")
        return
    if not MAIN_CHANNEL_ID:
        logger.warning("MAIN_CHANNEL_ID environment variable not set. Auto-summary feature disabled.")
        # Bot can continue for manual commands in discussion groups

    # Create the Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Initialize the application first to fetch bot details
    await application.initialize()
    logger.info("Application initialized.")

    # Get bot ID for filtering its own messages *after* initialization
    bot_id = application.bot.id
    logger.info(f"Bot ID: {bot_id}")

    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("explain", explain_message))
    application.add_handler(CommandHandler("summarise", summarise_message))

    # Add handler for new channel posts
    application.add_handler(MessageHandler(
        filters.ChatType.CHANNEL & (filters.TEXT | filters.CAPTION | filters.PHOTO) & filters.UpdateType.CHANNEL_POST,
        auto_summary
    ))

    # Add handler for forwarded messages in discussion groups
    # Note: This filter assumes discussion groups are SUPERGROUPS or standard GROUPS.
    # It also relies on the bot *not* having privacy mode enabled to see all messages.
    application.add_handler(MessageHandler(
        filters.Chat(chat_id=[int(gid) for gid in DISCUSSION_GROUP_IDS if gid.startswith('-')]) # Filter for specific group IDs
        & Filters.UpdateType.MESSAGE
        & Filters.FORWARDED,
        handle_discussion_forward
    ))

    # Add handler for anonymous comments in discussion groups
    # This should come AFTER specific handlers like commands and forwards
    # It filters for messages in the discussion groups that are NOT commands, NOT forwards, and NOT from the bot itself.
    application.add_handler(MessageHandler(
        Filters.Chat(chat_id=[int(gid) for gid in DISCUSSION_GROUP_IDS if gid.startswith('-')]) # Specific groups
        & Filters.UpdateType.MESSAGE                                                          # Regular messages
        & ~Filters.COMMAND                                                                    # Exclude commands
        & ~Filters.FORWARDED                                                                  # Exclude forwards
        & ~Filters.User(user_id=bot_id),                                                      # Exclude bot's own messages
        handle_anonymous_comment
    ))


    # Start the bot components
    logger.info("Starting bot polling...")
    await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    await application.start()
    logger.info("Bot started and polling.")

    # Keep the script running until interrupted (e.g., Ctrl+C)
    await asyncio.Future() # This will wait indefinitely


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
