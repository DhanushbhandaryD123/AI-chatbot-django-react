import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

function Chat() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleAsk = async () => {
    if (!question.trim()) return;

    const userMsg = { role: "user", text: question };
    setMessages(prev => [...prev, userMsg]);
    setQuestion("");
    setLoading(true);

    try {
      const res = await fetch(`${process.env.REACT_APP_API_URL}/api/chat/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMsg.text }),
      });

      const data = await res.json();

      setMessages(prev => [
        ...prev,
        { role: "bot", text: data.answer || "No answer found" }
      ]);
    } catch {
      setMessages(prev => [
        ...prev,
        { role: "bot", text: "Server error" }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  };

  return (
    <div className="chat-section">
      <h3>💬 Chat with PDF</h3>

      <div className="chat-window">
        <AnimatePresence initial={false}>
          {messages.map((msg, i) => (
            <motion.div
              key={i}
              className={`bubble ${msg.role}`}
              initial={{ opacity: 0, y: 16, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
            >
              {msg.text}
            </motion.div>
          ))}
        </AnimatePresence>

        {loading && (
          <motion.div
            className="bubble bot typing"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
          >
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
          </motion.div>
        )}
      </div>

      <div className="input-row">
        <textarea
          rows="2"
          placeholder="Ask something from the PDF..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
        />

        <motion.button
          onClick={handleAsk}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          ➤ Ask
        </motion.button>
      </div>
    </div>
  );
}

export default Chat;
