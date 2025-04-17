import React, { useState } from "react";
import "./TrueFalseExercise.css"; // Make sure to import the CSS file

const TrueFalseExercise = () => {
  const questions = [
    {
      question:
        "PyTorch tensors can be converted to NumPy arrays and vice versa.",
      answer: true,
    },
    {
      question:
        "In PyTorch, tensor shapes can be changed using the reshape() function.",
      answer: true,
    },
    {
      question:
        "Google Colab offers free access to GPUs including Nvidia Tesla T4.",
      answer: true,
    },
    {
      question: "The default data type of tensors in PyTorch is float64.",
      answer: false,
    },
    {
      question:
        "You can check if you have GPU access in Google Colab by using the nvidia-smi command.",
      answer: true,
    },
    {
      question:
        "PyTorch's torch.stack() function can be used to add a dimension to a tensor.",
      answer: false,
    },
    {
      question: "You can change the shape of a tensor using the view() method.",
      answer: true,
    },
    {
      question:
        "In PyTorch, tensor types like float32 are faster than float64 on GPUs.",
      answer: true,
    },
    {
      question:
        "The numpy() method on a PyTorch tensor returns a NumPy array with the same dtype and shape.",
      answer: true,
    },
    { question: "PyTorch tensors always reside on the GPU.", answer: false },
  ];

  const [answers, setAnswers] = useState(
    new Array(questions.length).fill(null)
  );
  const [correctAnswers, setCorrectAnswers] = useState(
    new Array(questions.length).fill(null)
  );

  const handleAnswerChange = (index, answer) => {
    const newAnswers = [...answers];
    newAnswers[index] = answer;
    setAnswers(newAnswers);
  };

  const handleSubmit = () => {
    const newCorrectAnswers = questions.map(
      (q, index) => answers[index] === q.answer
    );
    setCorrectAnswers(newCorrectAnswers);
  };

  return (
    <div className="exercise-container">
      <h1 className="exercise-title">True or False Exercise</h1>
      <form>
        {questions.map((q, index) => (
          <div key={index} className="question-container">
            <p className="question">{q.question}</p>
            <div className="options">
              <label className="radio-label">
                <input
                  type="radio"
                  name={`question-${index}`}
                  value="true"
                  checked={answers[index] === true}
                  onChange={() => handleAnswerChange(index, true)}
                  className="radio-input"
                />
                <span className="radio-text">True</span>
              </label>
              <label className="radio-label">
                <input
                  type="radio"
                  name={`question-${index}`}
                  value="false"
                  checked={answers[index] === false}
                  onChange={() => handleAnswerChange(index, false)}
                  className="radio-input"
                />
                <span className="radio-text">False</span>
              </label>
            </div>

            {correctAnswers[index] !== null &&
              answers[index] !== correctAnswers[index] && (
                <p className="feedback incorrect">
                  Incorrect! Correct answer is: {q.answer ? "True" : "False"}
                </p>
              )}
            {correctAnswers[index] !== null &&
              answers[index] === correctAnswers[index] && (
                <p className="feedback correct">Correct! 🎉</p>
              )}
          </div>
        ))}
      </form>

      <button className="submit-button" onClick={handleSubmit}>
        Submit
      </button>
    </div>
  );
};

export default TrueFalseExercise;
