document.addEventListener("DOMContentLoaded", () => {
    const messagesContainer = document.getElementById("chat-messages");
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");

    sendButton.addEventListener("click", sendMessage);

    userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            sendMessage();
            userInput.value = "";
        }
    });

    async function sendMessage() {
        const userMessage = userInput.value.trim();
        if (!userMessage) return;

        addMessage(userMessage, "user");

        try {
            const response = await fetch("/sentiment/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-CSRFToken": getCookie('csrftoken')
                },
                body: JSON.stringify(
                    { 
                        text: userMessage 
                    }
                )
            });

            if (!response.ok) {
                throw new Error("Error fetching response");
            }

            const data = await response.json();
            const botMessage = `${data.sentiment_description}: ${data.response}`;

            setTimeout(() => {
                addMessage(botMessage, "bot");
            }, 500);
            
        } catch (error) {
            console.error("Error:", error);
            addMessage("Sorry, something went wrong. Please try again later.", "bot");
        }

        userInput.value = ""; 
    }

    function addMessage(text, sender) {
        const message = document.createElement("div");
        message.className = `message ${sender}-message`;
        message.textContent = text;
        messagesContainer.appendChild(message);

        scrollToBottom();
    }

    function scrollToBottom() {
        const chatBody = document.getElementById("chat-body");
        chatBody.scrollTop = chatBody.scrollHeight;
    }

    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
});