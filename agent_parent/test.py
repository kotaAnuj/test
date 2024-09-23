import os
import tensorflow as tf
import openai

class AIAgent:
    def __init__(self):
        self.model = self._build_model()
        api_key = "2d2d8893395640a89cf366271e1a84fe"
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.aimlapi.com")

    def query_llm(self, prompt, temperature=0.7, max_tokens=150):
        try:
            response = self.client.chat.completions.create(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return None

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(4, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict(self, input_data):
        if input_data.shape[1] != 10:
            raise ValueError("Input shape must be (n, 10)")
        return self.model.predict(input_data)

agent = AIAgent()    
prompt = "Explain the concept of machine learning in simple terms."
response = agent.query_llm(prompt)
print(f"LLM Response: {response}")