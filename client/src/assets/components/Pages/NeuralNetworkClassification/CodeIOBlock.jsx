import React, { useState } from "react";
import { FaCopy, FaSun, FaMoon } from "react-icons/fa";
import "./CodeBlock.css";

const CodeIOBlock = ({ inputCode, outputCode }) => {
  const [isDarkMode, setIsDarkMode] = useState(false);

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  const copyToClipboard = (code) => {
    navigator.clipboard.writeText(code).then(() => {
      alert("Code copied!");
    });
  };

  return (
    <div className={`code-block-container ${isDarkMode ? "dark" : "light"}`}>
      <button
        className="copy-btn"
        title="Copy code"
        onClick={() => copyToClipboard(inputCode)}
      >
        <FaCopy />
      </button>

      <button
        className="theme-toggle-btn"
        title="Toggle Theme"
        onClick={toggleTheme}
      >
        {isDarkMode ? <FaSun /> : <FaMoon />}
      </button>

      {/* Input Code Block */}
      <div className="code-section">
        <div className="code-label">Input</div>
        <pre>
          <code>{inputCode}</code>
        </pre>
      </div>

      {/* Output Code Block */}
      {outputCode && (
        <div className="code-section">
          <div className="code-label">Output</div>
          <pre>
            <code>{outputCode}</code>
          </pre>
        </div>
      )}
    </div>
  );
};

export default CodeIOBlock;
