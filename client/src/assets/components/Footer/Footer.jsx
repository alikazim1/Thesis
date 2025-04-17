import React from "react";
import { FaInstagram, FaWhatsapp, FaYoutube } from "react-icons/fa";
import { TbWorldWww } from "react-icons/tb";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";

import Services from "../Services/Services.jsx";
import Hero from "../Hero/Hero.jsx";
import Banner from "../Banner/Banner.jsx";
import Subscribe from "../Subscribe/Subscribe.jsx";
import Banner2 from "../Banner/Banner2.jsx";

import Fundamentals from "../Pages/Fundamentals/Fundamentals";
import Workflow from "../Pages/WorkflowFundamentals/workflowFundamentals";
import ComputerVision from "../Pages/ComputerVision/ComputerVison";
import NeuralNetworksClassification from "../Pages/NeuralNetworkClassification/neuralNetworkClassification";
import CustomDatasets from "../Pages/CustomDatasets/customDataSet";
import GoingModular from "../Pages/GoingModular/GoingModular";
import TransferLearning from "../Pages/TransferLearning/transferLearning";

const Footer = () => {
  return (
    <footer className="py-28 bg-[#f7f7f7]">
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        whileInView={{ opacity: 1, y: 0 }}
        className="container"
      >
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-14 md:gap-4">
          {/* first section */}
          <div className="space-y-4 max-w-[300px]">
            <h1 className="text-2xl font-bold">PyTorch Learning Journey</h1>
            <p className="text-dark2">
              Dive into deep learning with our PyTorch Learning Journey. From
              beginner-friendly introductions to advanced model-building
              techniques, we offer a hands-on experience designed to help you
              master PyTorch, build real-world AI projects, and accelerate your
              career in machine learning.
            </p>
          </div>
          {/* second section */}
          <div className="grid grid-cols-2 gap-10">
            <div className="space-y-4">
              <h1 className="text-2xl font-bold">Courses</h1>
              <div className="text-dark2">
                <ul className="space-y-2 text-lg">
                  <li className="cursor-pointer hover:text-secondary duration-200">
                    <Link to="/fundamentals">Fundamentals of PyTorch</Link>
                  </li>
                  <li className="cursor-pointer hover:text-secondary duration-200">
                    <Link to="/workflowFundamentals">
                      Workflow Fundamentals
                    </Link>
                  </li>
                  <li className="cursor-pointer hover:text-secondary duration-200">
                    <Link to="/ComputerVison">Computer Vision</Link>
                  </li>
                  <li className="cursor-pointer hover:text-secondary duration-200">
                    <Link to="/customDataSet">Custom Datasets</Link>
                  </li>
                  <li className="cursor-pointer hover:text-secondary duration-200">
                    <Link to="/GoingModular">Going Modular</Link>
                  </li>
                  <li className="cursor-pointer hover:text-secondary duration-200">
                    <Link to="/transferLearning">Transfer Learning</Link>
                  </li>
                </ul>
              </div>
            </div>
          </div>
          {/* third section */}
          <div className="space-y-4 max-w-[300px] -ml-20">
            <h1 className="text-2xl font-bold">Get In Touch</h1>
            <div className="flex items-center">
              <input
                type="text"
                placeholder="Enter your email"
                className="p-3 rounded-s-xl bg-white w-full py-4 focus:ring-0 focus:outline-none placeholder:text-dark2"
              />
              <button className="bg-primary text-white font-semibold py-4 px-6 rounded-e-xl">
                Go
              </button>
            </div>
            {/* social icons */}
            <div className="flex space-x-6 py-3">
              <a href="https://chat.whatsapp.com/Fs5vUHa8VvnLnuzv3qY9Ee">
                <FaWhatsapp className="cursor-pointer hover:text-primary hover:scale-105 duration-200" />
              </a>
              <a href="">
                <FaInstagram className="cursor-pointer hover:text-primary hover:scale-105 duration-200" />
              </a>
              <a href="">
                <TbWorldWww className="cursor-pointer hover:text-primary hover:scale-105 duration-200" />
              </a>
              <a href="">
                <FaYoutube className="cursor-pointer hover:text-primary hover:scale-105 duration-200" />
              </a>
            </div>
          </div>
        </div>
      </motion.div>
    </footer>
  );
};

export default Footer;
