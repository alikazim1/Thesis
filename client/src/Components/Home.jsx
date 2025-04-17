// Home.jsx
import React from "react";
import { useLocation } from "react-router-dom";
import AppLayout from "./AppLayout"; // adjust path if needed

export const Home = () => {
  const location = useLocation();
  const user = location.state?.user;

  return (
    <>
      <h1 className="text-4xl font-bold text-center text-gray-800 py-6 bg-gradient-to-r from-pink-500 via-red-500 to-yellow-500 bg-clip-text text-transparent shadow-md">
        {" "}
        Welcome Home <span className="text-black">
          {user?.name || "Guest"}
        </span>{" "}
        Happy PyTorch Learning
      </h1>
      <AppLayout />
    </>
  );
};
