import langchain_core
import langchain_openai
from query_llm import LLMQueryer

sys_message = "You are a helpful assistant who likes to play along and respond to a child asking jokes. Any question you ask should end in three question marks."

test_message = [
    {
      "role": "user",
      "content": [{ "type": "text", "text": "knock knock." }]
    },
    {
      "role": "assistant",
      "content": [{ "type": "text", "text": "Who's there?" }]
    },
    {
      "role": "user",
      "content": [{ "type": "text", "text": "Orange." }]
    }
  ]

llmqueryer = LLMQueryer(project_name='default',connection_type='local')
for response in llmqueryer.query_llm(sys_msg=sys_message,human_msg=test_message,include_rag=False):
    print(response)
    print("yayo!")