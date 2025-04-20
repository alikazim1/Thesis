import { delay } from "framer-motion";
import React from "react";
import { Link } from "react-router-dom";
import { AiTwotoneExperiment } from "react-icons/ai";
import { BiCodeAlt } from "react-icons/bi"; // For Fundamentals
import { BiMerge } from "react-icons/bi"; // For Workflow Fundamentals
import { BiCamera } from "react-icons/bi"; // For Computer Vision
import { BiBarChartAlt } from "react-icons/bi"; // For Custom Datasets
// import { BiCogs } from "react-icons/bi"; // For Going Modular
import { BiTransferAlt } from "react-icons/bi"; // For Transfer Learning
import { motion } from "framer-motion";
import { FaDatabase } from "react-icons/fa";
import { VscSymbolClass } from "react-icons/vsc";

import { PiShareNetworkFill } from "react-icons/pi";

const ServicesData = [
  {
    id: 1,
    title: "Fundamentals",
    link: "/fundamentals",

    icon: <BiCodeAlt />,
    delay: 0.2,
  },
  {
    id: 2,
    title: "Workflow Fundamentals",
    link: "/workflow",
    icon: <BiMerge />,
    delay: 0.3,
  },
  {
    id: 3,
    title: "Neural Networks Classification",
    link: "/neuralNetworksClassification",
    icon: <PiShareNetworkFill />,
    delay: 0.4,
  },
  {
    id: 4,
    title: "Computer Vision",
    link: "/computerVision",
    icon: <BiCamera />,
    delay: 0.5,
  },
  {
    id: 5,
    title: "Custom Datasets",
    link: "/customDatasets",
    icon: <FaDatabase />,
    delay: 0.6,
  },

  {
    id: 6,
    title: "Going Modular",
    link: "/goingModular",
    icon: <VscSymbolClass />,
    delay: 0.7,
  },
  {
    id: 7,
    title: "Transfer Learning",
    link: "/transferLearning",
    icon: <BiTransferAlt />,
    delay: 0.7,
  },
  {
    id: 8,
    title: "Experiment Tracking",
    link: "/experimentTracking",
    icon: <AiTwotoneExperiment />,
    delay: 0.7,
  },
];

const SlideLeft = (delay) => {
  return {
    initial: {
      opacity: 0,
      x: 50,
    },
    animate: {
      opacity: 1,
      x: 0,
      transition: {
        duration: 0.3,
        delay: delay,
        ease: "easeInOut",
      },
    },
  };
};

const Service = () => {
  return (
    <section className="bg-white">
      <div className="container pb-14 pt-16">
        <h1 className="text-4xl font-bold text-left pb-10">
          The Included Sections
        </h1>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-8">
          {ServicesData.map((item) => (
            <Link to={item.link} key={item.id}>
              <motion.div
                variants={SlideLeft(item.delay)}
                initial="initial"
                whileInView={"animate"}
                viewport={{ once: true }}
                className="bg-[#f4f4f4] rounded-2xl flex flex-col gap-4 items-center justify-center p-4 py-7 hover:bg-white hover:scale-110 duration-300 hover:shadow-2xl"
              >
                <div className="text-4xl mb-4">{item.icon}</div>
                <h1 className="text-lg font-semibold text-center px-4">
                  {item.title}
                </h1>
              </motion.div>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Service;
