import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import AuthService from "../services/AuthService"; // ✅ Add this at the top
import { Login } from "./Components/Login";
import { Signup } from "./Components/Signup";
import { Home } from "./Components/Home";
import { Navbar } from "./Components/Navbar";
import AppLayout from "./Components/AppLayout"; // ✅ Import this
import { useEffect, useState, createContext } from "react";
import axios from "axios";
import "./index.css"; // ✅ Import your CSS file here

export const IsLoggedInContext = createContext();
export const SetIsLoggedInContext = createContext();

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  // 00p here
  useEffect(() => {
    AuthService.getUser().then((data) => {
      setIsLoggedIn(!!data?.user);
    });
  }, []);

  return (
    <SetIsLoggedInContext.Provider value={setIsLoggedIn}>
      <IsLoggedInContext.Provider value={isLoggedIn}>
        <BrowserRouter>
          <Navbar isLoggedIn={isLoggedIn} setIsLoggedIn={setIsLoggedIn} />
          <Routes>
            <Route path="/home" element={<Home />} />
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={<Signup />} />

            {/* ✅ Add this to enable chapters/pages */}
            <Route path="/*" element={<AppLayout />} />
          </Routes>
        </BrowserRouter>
      </IsLoggedInContext.Provider>
    </SetIsLoggedInContext.Provider>
  );
}

export default App;
