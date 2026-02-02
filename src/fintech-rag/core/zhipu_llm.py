import os

from dotenv import load_dotenv
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from langchain.messages import AIMessage, HumanMessage, SystemMessage
load_dotenv()
class ZhipuAILLM:
    def __init__(self, temperature=0.7, model="glm-4"):
        
        """
        Initialize Zhipu AI LLM client
        """
        api_key = os.getenv("ZHIPUAI_API_KEY")
        if not api_key:
            raise ValueError("ZHIPUAI_API_KEY not found in environment variables")
        self.llm = ChatZhipuAI(
            api_key=api_key,
            temperature=temperature,
            model=model
        )
    def chat(self, user_message, system_message=None, print_output=True):
        """
        Send messages to Zhipu AI and get response
        """
        messages = []
        messages.append(AIMessage(content="Hi."))
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=user_message))

        try:
            response = self.llm.invoke(messages)

            if print_output:
                self._print_conversation(user_message, response.content)
            return response.content
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if print_output:
                print(f"{error_msg}")
            return error_msg
        
    def _print_conversation(self, user_msg, ai_response):
        """
        Format and print the conversation
        """
        print("\n" + "="*60)
        print(f"USER:{user_msg}")
        print("-"*60)
        print(f"AI:{ai_response}")
        print("="*60 + "\n")

if __name__ == "__main__":
    llm = ZhipuAILLM(temperature=0.7)
    llm.chat(user_message="Write a short poem about AI in four lines.", system_message="Your role is a poet")