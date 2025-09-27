from __future__ import annotations

from dataclasses import dataclass

import asyncio
from math import log
import os
from urllib.parse import quote

from dotenv import load_dotenv
from telebot.async_telebot import AsyncTeleBot
from telebot.types import Message, BotCommand
import logfire

from vibe_emp.guess import (
    OpenAIChatModel,
    guess as run_guess_round,
    init_llm,
)
from vibe_emp.gameplay import (
    choose_document,
    advance_game_round,
)


@dataclass(slots=True, kw_only=True)
class GameSession:
    chat_id: int
    document: str
    doc_name: str
    rounds: list[tuple[str, str]]
    last_bot_message_id: int | None
    last_hint_text: str | None
    user_guess_log: list[str]


_bot: AsyncTeleBot | None = None
_llm: OpenAIChatModel | None = None
_sessions: dict[int, GameSession] = {}


async def _ensure_llm() -> OpenAIChatModel:
    global _llm
    if _llm is None:
        _llm = init_llm()
    return _llm


async def _choose_document() -> tuple[str, str]:
    document, doc_name = choose_document()
    return document, doc_name


async def _start_game(chat_id: int) -> None:
    document, doc_name = await _choose_document()
    session = GameSession(
        chat_id=chat_id,
        document=document,
        doc_name=doc_name,
        rounds=[],
        last_bot_message_id=None,
        last_hint_text=None,
        user_guess_log=[],
    )
    _sessions[chat_id] = session


async def _advance_game_from_reply(message: Message) -> tuple[str, bool]:
    chat_id = message.chat.id
    session = _sessions.get(chat_id)
    if session is None:
        return "No active game. Use /guess to start one.", False

    user_guess = (message.text or "").strip()
    if not user_guess:
        return "Please reply with text to continue the game.", False

    if session.last_bot_message_id is None:
        return "Game state error. Use /q then /guess to restart.", False

    llm = await _ensure_llm()
    try:
        text, finished, updated_rounds, new_last_hint = await advance_game_round(
            llm=llm,
            document=session.document,
            rounds=session.rounds,
            user_guess=user_guess,
            last_hint_text=session.last_hint_text,
        )
        session.rounds = updated_rounds
        session.last_hint_text = new_last_hint
    except Exception:
        _sessions.pop(chat_id, None)
        return "Internal error. The game was stopped. Use /guess to start again.", True

    if finished:
        del _sessions[chat_id]
        url_name = quote(session.doc_name, safe="()/._-~%")
        return text + f"\n\nhttps://zh.wikipedia.org/wiki/{url_name}", True

    return text, False


async def create_bot() -> AsyncTeleBot:
    global _bot
    if _bot is not None:
        return _bot

    load_dotenv()
    token = os.getenv("TG_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing TG_BOT_TOKEN in environment")

    bot = AsyncTeleBot(token)

    @bot.message_handler(commands=["ping"])
    async def handle_ping(message: Message) -> None:
        await bot.reply_to(message, str(message.chat.id))

    @bot.message_handler(commands=["guess"])
    async def handle_guess(message: Message) -> None:
        chat_id = message.chat.id
        if chat_id in _sessions:
            await bot.reply_to(message, "a thread existing")
            return
        # send placeholder first
        placeholder = await bot.send_message(chat_id, "…", reply_to_message_id=message.message_id)
        try:
            await _start_game(chat_id)
            session = _sessions.get(chat_id)
            if session is None:
                await bot.edit_message_text("Internal error. Please try again.", chat_id, placeholder.message_id)
                return
            llm = await _ensure_llm()
            kind, content = await run_guess_round(llm=llm, document=session.document, rounds=session.rounds)
            # edit placeholder with first hint
            await bot.edit_message_text(content, chat_id, placeholder.message_id)
            session.last_bot_message_id = placeholder.message_id
            session.last_hint_text = content
        except Exception:
            _sessions.pop(chat_id, None)
            await bot.edit_message_text("Internal error starting the game. Please try /guess again.", chat_id, placeholder.message_id)
            return

    @bot.message_handler(commands=["q"])
    async def handle_quit(message: Message) -> None:
        chat_id = message.chat.id
        if chat_id in _sessions:
            session = _sessions.pop(chat_id)
            url_name = quote(session.doc_name, safe="()/._-~%")
            await bot.reply_to(message, f"the answer is: {session.doc_name}\nhttps://zh.wikipedia.org/wiki/{url_name}")
        else:
            await bot.reply_to(message, "No active game to quit.")

    @bot.message_handler(content_types=["text"]) 
    async def handle_reply(message: Message) -> None:
        chat_id = message.chat.id
        session = _sessions.get(chat_id)
        if session is None:
            return
        if message.reply_to_message is None:
            return
        if message.reply_to_message.from_user is None or message.reply_to_message.from_user.is_bot is not True:
            return
        if session.last_bot_message_id is None or message.reply_to_message.message_id != session.last_bot_message_id:
            return

        # send placeholder reply first
        placeholder = await bot.send_message(chat_id, "…", reply_to_message_id=message.message_id)
        text, finished = await _advance_game_from_reply(message)
        # Build aggregated text by interleaving all previous hints with stored guesses, then the new guess and hint
        aggregated_text_parts: list[str] = []
        if chat_id in _sessions:
            session = _sessions[chat_id]
            previous_hints: list[str] = [note for note, _ in session.rounds]
            previous_guesses: list[str] = session.user_guess_log
            for idx, hint_text in enumerate(previous_hints):
                aggregated_text_parts.append(hint_text)
                if idx < len(previous_guesses):
                    aggregated_text_parts.append(previous_guesses[idx])
        # Format and store the user's display name and guess
        from_user = message.from_user
        display_name: str = (
            (from_user.username if from_user and from_user.username else None)
            or (
                (" ".join(filter(None, [from_user.first_name if from_user else None, from_user.last_name if from_user else None])).strip())
                if from_user else None
            )
            or "user"
        )
        user_guess: str = (message.text or "").strip()
        formatted_guess: str = f"{display_name}: {user_guess}"
        if chat_id in _sessions:
            _sessions[chat_id].user_guess_log.append(formatted_guess)
        # Append the current round's guess and the newly generated hint
        aggregated_text_parts.append(formatted_guess)
        aggregated_text_parts.append(text)
        aggregated_text: str = "\n\n".join(aggregated_text_parts)

        await bot.edit_message_text(aggregated_text, chat_id, placeholder.message_id)

        if not finished and chat_id in _sessions:
            # Delete the previous hint message to keep only the latest aggregated one
            previous_message_id: int | None = _sessions[chat_id].last_bot_message_id
            if previous_message_id is not None:
                try:
                    await bot.delete_message(chat_id, previous_message_id)
                except Exception:
                    pass

            # Update last bot message id and last hint text for the next reply
            _sessions[chat_id].last_bot_message_id = placeholder.message_id
            # Keep only the latest single hint here for LLM rounds context
            _sessions[chat_id].last_hint_text = text
        else:
            # Clear any lingering state just in case
            _sessions.pop(chat_id, None)

    await bot.set_my_commands([
        BotCommand("guess", "start a game thread"),
        BotCommand("q", "quit current game and reveal answer"),
        BotCommand("ping", "return current chat id"),
    ])

    _bot = bot
    return bot


async def main() -> None:
    logfire.configure()
    logfire.instrument_pydantic_ai()

    bot = await create_bot()
    await bot.polling(non_stop=True, request_timeout=60)


if __name__ == "__main__":
    asyncio.run(main())
