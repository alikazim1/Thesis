// Function to copy code content to clipboard
function copyCode(button) {
    const codeBlock = button.closest('.code-block').querySelector('code');
    const codeText = codeBlock.textContent || codeBlock.innerText;
  
    // Create a temporary textarea to copy the text from
    const textarea = document.createElement('textarea');
    textarea.value = codeText;
    document.body.appendChild(textarea);
    textarea.select(); // Select the text
    document.execCommand('copy'); // Execute the copy command
    document.body.removeChild(textarea); // Remove the textarea
  
    // Optionally, change button text or style to show it was copied
    button.innerText = "Copied!";
    setTimeout(() => {
      button.innerText = "Copy";
    }, 1500); // Reset the button text after 1.5 seconds
  }
  