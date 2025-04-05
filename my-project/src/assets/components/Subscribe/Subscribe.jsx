import React from "react";
import { FaBell } from "react-icons/fa";
import { motion } from "framer-motion";
import BgImage from "../../../assets/bg.png";

const bgStyle = {
  backgroundImage: `url(${BgImage})`,
  backgroundSize: "cover",
  backgroundrepeat: "no-repeat",
  backgroundposition: "center",
};

const Subscribe = () => {
  return (
    <section className="bg-[#f7f7f7]">
      <motion.div
        whileInView={{ opacity: 1 }}
        style={bgStyle}
        className="container py-24 md:py-48"
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.5 }}
          whileInView={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, ease: "easeInOut" }}
          className="flex flex-col justify-center"
        >
          <div className="text-center space-y-4 lg:max-w-[430px] mx-auto">
            <h1 className="text-4xl font-bold !leading-snug">
              4K+ Students Enrolled and Learning from Us
            </h1>
            <p>
              Everystudent is satisfied with our courses and learning
              experience. Join us today and unlock your potential with our
              Courses.
            </p>
            <a
              href=""
              className="primary-btn !mt-8 inline-flex items-center gap-4 group"
            >
              Subscribe Now
              <FaBell className="group-hover:animate-bounce group-hover:text-lg duration-200" />
            </a>
          </div>
        </motion.div>
      </motion.div>
    </section>
  );
};

export default Subscribe;
