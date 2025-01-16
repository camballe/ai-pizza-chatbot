document.addEventListener("DOMContentLoaded", () => {
  const userInput = document.getElementById("user-input");
  const sendButton = document.getElementById("send-button");
  const chatBox = document.getElementById("chat-box");

  function createLoadingIndicator() {
    const loading = document.createElement("div");
    loading.className = "loading";
    loading.innerHTML = `
            <span></span>
            <span></span>
            <span></span>
        `;
    return loading;
  }

  function appendMessage(message, isUser) {
    const messageDiv = document.createElement("div");
    messageDiv.className = isUser ? "user-message" : "bot-message";

    // Check if previous message was from same sender
    const lastMessage = chatBox.lastElementChild;
    if (
      lastMessage &&
      ((isUser && lastMessage.classList.contains("user-message")) ||
        (!isUser && lastMessage.classList.contains("bot-message")))
    ) {
      messageDiv.classList.add("grouped");
    }

    messageDiv.textContent = message;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  function sendMessage() {
    const message = userInput.value.trim();
    if (message) {
      appendMessage(message, true);
      userInput.value = "";

      // Add loading indicator
      const loadingIndicator = createLoadingIndicator();
      chatBox.appendChild(loadingIndicator);
      chatBox.scrollTop = chatBox.scrollHeight;

      fetch("/get", {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: `msg=${encodeURIComponent(message)}`,
      })
        .then((response) => response.json())
        .then((data) => {
          // Remove loading indicator
          console.log(data)
          loadingIndicator.remove();
          appendMessage(data.response, false);
        })
        .catch((error) => {
          console.error("Error:", error);
          loadingIndicator.remove();
          appendMessage("Sorry, something went wrong.", false);
        });
    }
  }

  sendButton.addEventListener("click", sendMessage);
  userInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      sendMessage();
    }
  });
});
