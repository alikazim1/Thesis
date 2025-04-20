import React from "react";
import { Routes, Route } from "react-router-dom";
import "../index.css"; // Import your CSS file here
//import Navbar from "../assets/components/Navbar/Navbar.jsx";
import Services from "../assets/components/Services/Services.jsx";
import Hero from "../assets/components/Hero/Hero.jsx";
import Banner from "../assets/components/Banner/Banner.jsx";
import Subscribe from "../assets/components/Subscribe/Subscribe.jsx";
import Banner2 from "../assets/components/Banner/Banner2.jsx";
import Footer from "../assets/components/Footer/Footer.jsx";
import Fundamentals from "../assets/components/Pages/Fundamentals/Fundamentals.jsx";
import Workflow from "../assets/components/Pages/WorkflowFundamentals/workflowFundamentals.jsx";
import ComputerVision from "../assets/components/Pages/ComputerVision/ComputerVison.jsx";
import NeuralNetworksClassification from "../assets/components/Pages/NeuralNetworkClassification/neuralNetworkClassification.jsx";
import CustomDatasets from "../assets/components/Pages/CustomDatasets/customDataSet.jsx";
import GoingModular from "../assets/components/Pages/GoingModular/GoingModular.jsx";
import TransferLearning from "../assets/components/Pages/TransferLearning/transferLearning.jsx";

const HomeLayout = () => (
  <>
    <Hero />
    <Services />
    <Banner />
    <Subscribe />
    <Banner2 />
    <Footer />
  </>
);

const AppLayout = () => {
  return (
    <main className="overflow-hidden bg-white text-dark">
      <Routes>
        <Route path="/" element={<HomeLayout />} />
        <Route path="/fundamentals" element={<Fundamentals />} />
        <Route path="/workflow" element={<Workflow />} />
        <Route
          path="/neuralNetworksClassification"
          element={<NeuralNetworksClassification />}
        />
        <Route path="/computerVision" element={<ComputerVision />} />
        <Route path="/customDatasets" element={<CustomDatasets />} />
        <Route path="/goingModular" element={<GoingModular />} />
        <Route path="/transferLearning" element={<TransferLearning />} />
      </Routes>
    </main>
  );
};

export default AppLayout;
