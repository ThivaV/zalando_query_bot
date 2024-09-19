from langchain_community.chat_models import ChatOllama

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class InitChat:
    def __init__(self, model_id: str = "gemma2:2b", keep_alive: str = "3h", max_tokens: int = 512, temperature: float = 0) -> None:
        self.model_id = model_id
        self.keep_alive = keep_alive
        self.max_tokens = max_tokens 
        self.temperature = temperature 

        self.model = ChatOllama(model= self.model_id, keep_alive=self.keep_alive, max_tokens=self.max_tokens, temperature=self.temperature)

    def talk_to_bot(self, question: str, context: str = None, chat_history: str = None):
        """talk to google gemma2 2b"""

        response = ""

        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        (
                            """
                            You are a helpful sales assistant AI for Zalando, a clothing e-commerce platform. 
                            Your goal is to assist customers with their queries about products, FAQs, and other general information. 

                            ### Instructions:
                            1. **Product Queries**: If the customer asks about a specific product or category, search the provided "Context" for suitable products.
                                - If relevant products are found, return their SKU codes in the following JSON format: {{"sku": ["sku1", "sku2", ...], "message": "We found these products that match your search!"}}.
                                - If no products are found, return the following message in the JSON format: {{"sku": [], "message": "We couldn't find any matching products. Please try rephrasing your query."}}.

                            2. **Non-Product Queries (FAQ)**: For FAQ-style questions (e.g., shipping, returns, etc.), provide a helpful and accurate response based on the context or knowledge base. If additional context is needed, politely ask for clarification.

                            3. **Chat History Consideration**: Always consider the previous conversation history to maintain context when answering follow-up questions or clarifications.

                            4. **Output Format**: Your response must strictly follow this JSON format: {{"sku": [], "message": ""}}. Always ensure the response is well-formatted, including correct SKU codes if applicable.

                            ### Inputs:
                            - **Context**: This contains available product information.
                            - **Customer Question**: The customer’s query.

                            ### Example Responses:
                            1. If products are found:
                                - Question: I'm looking for XYZ
                                - Response:
                                    ```json
                                    {{
                                        "sku": ["12345", "67890"],
                                        "message": "We found these products that match your search!"
                                    }}```
                            2. If no products are found:
                                - Question: Hi, you have ABC ?
                                - Response:
                                    ```json
                                    {{
                                        "sku": [],
                                        "message": "We couldn't find any matching products. Please try rephrasing your query."
                                    }}```
                            3. For FAQ-style questions:
                                - Question: What is Zalando ?
                                - Response:
                                    ```json
                                    {{
                                        "sku": [],
                                        "message": "Zalando is a European online fashion and lifestyle platform founded in 2008 in Berlin, Germany."
                                    }}```

                            ### Key Points:
                            - **Clear Instructions**: Detailed and specific instructions for handling product queries, FAQ queries, and how to structure the responses.
                            - **JSON Format**: The output is strictly enforced to follow the {{"sku": [], "message": ""}} format.
                            - **Context Use**: Ensures the assistant uses both the current context and chat history to answer the customer's questions accurately.

                            Context: {context}.
                            Chat history: {chat_history}
                            """
                        ),
                    ),
                    ("human", "{input}"),
                ]
            )

            chain = prompt | self.model
            response = chain.invoke(
                {
                    "context": context,
                    "chat_history": chat_history,
                    "input": chat_history,
                }
            )          
            
            print("\n")
            print("question: ", question)
            print("\n")
            print("context: ", context)
            print("\n")
            print("chat_history: ", chat_history)
            print("\n")
            print("response: ", response)

        except Exception as e:
            print(f"An error occured at talk_to_bot(): ", e)
            response = {"sku": [], "message": "Apologies, but I'm unable to respond to your queries at the moment. Please try again later."}

        return response