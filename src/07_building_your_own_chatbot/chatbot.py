# Youtube Streamlit Playlist

# https://www.youtube.com/watch?v=hff2tHUzxJM&list=PLc2rvfiptPSSpZ99EnJbH5LjTJ_nOoSwW

import streamlit as st

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


from langchain_core.runnables import RunnableWithMessageHistory


from langchain_core.prompts import (
	ChatPromptTemplate,
	HumanMessagePromptTemplate,
	MessagesPlaceholder,
	SystemMessagePromptTemplate,
)

from langchain_community.chat_message_histories import SQLChatMessageHistory


class LLMBackend:
	def __init__(self, session_id: str):
		load_dotenv(".env")
		base_url = os.getenv("OLLAMA_BASE_URL")
		model = os.getenv("OLLAMA_LLM_MODEL")
		self.llm = ChatOllama(base_url=base_url, model=model)
		self.session_id = session_id
		self.chain = self._configure_chain()
		self._runnable_with_history = self._configure_runnable_with_history()

	def chat(self, user_input: str):
		return self._runnable_with_history.invoke(
			{"input": user_input},
			config={"configurable": {"session_id": self.session_id}},
		)

	def _configure_chain(self):
		prompt = ChatPromptTemplate.from_messages(
				[
						SystemMessagePromptTemplate.from_template(
								"You are a helpful assistant."
						),
						MessagesPlaceholder("history"),
						HumanMessagePromptTemplate.from_template("{input}"),
				]
		)

		return prompt | self.llm | StrOutputParser()


	def _configure_runnable_with_history(self):
		return RunnableWithMessageHistory(
			self.chain,
			self._get_session_history,
			input_messages_key="input",
			history_messages_key="history",
		)

	def _get_session_history(self, session_id: str):
		return SQLChatMessageHistory(
			session_id=session_id, connection_string="sqlite:///chat_history.db"
		)


class Chatbot:
	def __init__(self, session_id: str):
		self.history: SQLChatMessageHistory | None = None
		self.backend = LLMBackend(session_id=session_id)
		self.session_id = session_id

	def start_app(self):
		st.title("Chatbot Application")
		st.write("This is a simple chatbot application built using LangChain and Ollama.")

		self.session_id = st.text_input("Enter your name:", self.session_id)

		if "chat_history" not in st.session_state:
			st.session_state.chat_history = []

		if st.button("Start New Conversation"):
			self.history = self.backend._get_session_history(self.session_id)
			self.history.clear()

		self._display_chat_history()

		self._display_text_input()

	def _display_chat_history(self):
		print(f"Displaying chat history {st.session_state.chat_history}")
		for message in st.session_state.chat_history:
			with st.chat_message(message["role"]):
				st.markdown(message["content"])

	def _display_text_input(self):
		prompt = st.chat_input("What's up?")
		if not prompt:
			return
		st.session_state.chat_history.append({"role": "user", "content": prompt})
		response = self.backend.chat(prompt)
		st.session_state.chat_history.append({"role": "assistant", "content": response})


def streamlit_app():
	Chatbot(session_id="my_session").start_app()


if __name__ == "__main__":
	streamlit_app()
