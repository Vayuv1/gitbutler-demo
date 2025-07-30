# We need to start dd_chat_server.py first and run this client.

import tkinter as tk
import requests

def send_prompt():
    user_input = entry.get().strip()
    if not user_input:
        return

    # Update UI immediately in main thread
    chat_text.config(state="normal")
    chat_text.insert(tk.END, f"You: {user_input}\n")
    entry.delete(0, tk.END)

    # Process API call synchronously (no threading)
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={"prompt": user_input},
            timeout=10
        )
        data = response.json()
        chat_text.insert(tk.END, f"AI: {data.get('response', '')}\n")
    except Exception as e:
        chat_text.insert(tk.END, f"Error: {str(e)}\n")

    chat_text.config(state="disabled")
    chat_text.see(tk.END)

# Minimal GUI setup
root = tk.Tk()
root.title("WSL Chat")

chat_text = tk.Text(root, wrap=tk.WORD, state="disabled")
chat_text.pack(expand=True, fill=tk.BOTH)

entry = tk.Entry(root)
entry.pack(fill=tk.X, padx=5, pady=5)
entry.bind("<Return>", lambda e: send_prompt())

tk.Button(root, text="Send", command=send_prompt).pack()

root.mainloop()