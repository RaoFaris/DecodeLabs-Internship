#Rule based AI-chatbot

#Using a dictionary instead of long if-elif ladder as it makes the lookups fast no matter how many rules we add as stated in the project file
#Knowledge base
responses = {
    "hello": "Hi there! How can i help you today?",
    "hi": "Hey! what can i do for you?",
    "how are you?": "Doing great. ",
    "what is your name?": "I'm cb1. ",
    "what can you do?": "I can respond and talk to you. Try saying 'hello' or 'bye'. ",
    "help": "Try saying 'hello', 'how are you' or 'what is your name'.",
    "bye": "Goodbye! Have a great day!",
    "quit": "Goodbye! Have a great day!",
    "exit": "Goodbye! Have a great day!",
}

#commands to end the loop
EXIT_COMMANDS = {"bye","exit","quit"}

def get_response(user_input: str) -> str:
    """Look up a cleaned user input in the knowledge base, with a fallback"""
    return responses.get(user_input, "I do not understand. Try 'help' to see what can i do.")

def run_chatbot():
    print("Chatbot: Hello! How can i help you today?")
    print("... and type 'bye', 'exit' or 'quit' if you want to end the chat.")

#input loop: runs until exit command breaks
    while True:
        raw_input_text = input("You: ")

        #Sanitization
        clean_input = raw_input_text.lower().strip()

        #Exit strategy
        if clean_input in EXIT_COMMANDS:
            print(f"Chatbot: {responses[clean_input]}")
            break
            
        reply = get_response(clean_input)
        print(f"Chatbot: {reply}")

if __name__ == "__main__":
    run_chatbot()