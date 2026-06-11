import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Chat from "./components/Chat";
import UploadPDF from "./components/UploadPDF";
import "./App.css";

function App() {
  const [dark, setDark] = useState(true);

  return (
    <div className={dark ? "page dark" : "page light"}>
      <div className="orb orb-1" />
      <div className="orb orb-2" />
      <div className="orb orb-3" />

      <motion.div
        className="app-container"
        initial={{ opacity: 0, y: 40, scale: 0.96 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
      >
        <div className="header-row">
          <motion.h1
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            🤖 AI PDF Chatbot
          </motion.h1>

          <motion.button
            className="theme-btn"
            onClick={() => setDark(!dark)}
            whileHover={{ scale: 1.08 }}
            whileTap={{ scale: 0.92, rotate: 15 }}
          >
            <AnimatePresence mode="wait" initial={false}>
              <motion.span
                key={dark ? "sun" : "moon"}
                initial={{ rotate: -90, opacity: 0 }}
                animate={{ rotate: 0, opacity: 1 }}
                exit={{ rotate: 90, opacity: 0 }}
                transition={{ duration: 0.25 }}
                style={{ display: "inline-block" }}
              >
                {dark ? "☀️ Light" : "🌙 Dark"}
              </motion.span>
            </AnimatePresence>
          </motion.button>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <UploadPDF />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <Chat />
        </motion.div>
      </motion.div>
    </div>
  );
}

export default App;
