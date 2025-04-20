import React from "react";
import Navbar from "../Navbar/Navbar";
import { IoIosArrowRoundForward } from "react-icons/io";
import Blob from "../../../assets/blob.svg";
import HeroPng from "../../../assets/hero.png";
import Robo from "../../../assets/robo.png";
import Torch from "../../../assets/torch.png";
import { motion } from "framer-motion";

export const FadeUp = (delay) => {
  return {
    initial: {
      opacity: 0,
      y: 50,
    },
    animate: {
      opacity: 1,
      y: 0,
      transition: {
        type: "spring",
        stiffness: 100,
        duration: 0.5,
        delay: delay,
        ease: "easeInOut",
      }, // Optionally apply delay to the transition
    },
    exit: {
      opacity: 0,
      y: 50,
    },
  };
};

const Hero = () => {
  return (
    <section className="bg-light overflow-hidden relative">
      <Navbar />
      <div
        className="container grid grid-cols-1
      md:grid-cols-2 min-h-[650px]"
      >
        {/* Brand Info */}
        <div className="flex flex-col justify-center py-14 md:py-0 relative z-20">
          <div className="text-center md:text-left space-y-10 lg:max-w-[400px]">
            <motion.h1
              variants={FadeUp(0.6)}
              initial="initial"
              animate="animate"
              className="text-3xl lg:text-5xl font-bold !leading-snug"
            >
              Let's learn{" "}
              <a
                href="https://pytorch.org"
                target="_blank"
                className="text-orange-500"
              >
                PyTorch
              </a>{" "}
              from the Beginning
            </motion.h1>
            <motion.div
              variants={FadeUp(0.8)}
              initial="initial"
              animate="animate"
              className="flex justify-center md:justify-start"
            ></motion.div>
          </div>
        </div>
        {/* Hero Image */}
        <div className="flex justify-center items-center relative">
          <motion.img
            initial={{ x: 50, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.4, ease: "easeInOut" }}
            src={Torch}
            alt="PyTorch Logo"
            className="w-[400px] xl:w-[600px] relative z-10 drop-shadow"
          />
          <motion.img
            src={Robo}
            alt="Robot"
            className="absolute w-[250px] md:w-[300px] z-[5]"
            style={{
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
            }}
          />
        </div>
      </div>
    </section>
  );
};

export default Hero;
