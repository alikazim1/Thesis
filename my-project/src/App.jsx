import React from "react";
import Navbar from "./assets/components/Navbar/Navbar.jsx";
import Services from "./assets/components/Services/Services.jsx";
import Hero from "./assets/components/Hero/Hero.jsx";
import Banner from "./assets/components/Banner/Banner.jsx";
import Subscribe from "./assets/components/Subscribe/Subscribe.jsx";
import Banner2 from "./assets/components/Banner/Banner2.jsx";
import Footer from "./assets/components/Footer/Footer.jsx";

const App = () => {
  return (
    <main className="overflow-hidden bg-white text-dark">
      <Hero />
      <Services />
      <Banner />
      <Subscribe />
      <Banner2 />
      <Footer />
    </main>
  );
};

export default App;
