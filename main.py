from dotenv import load_dotenv

load_dotenv()
from graph.graph import app
if __name__ == "__main__":
    print("Hello RAG")
    print(app.invoke(input={"question": "What does the medium is the message mean?"}))
    # What is the relationship between the case of Tony Timpa and of George Floyd?
    # What does the medium is the message mean?
    
     