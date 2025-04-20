import React, { useContext } from "react";
import { Button } from "@mui/material";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { SetIsLoggedInContext } from "../App";
export const Logout = () => {
  const navigate = useNavigate();
  const SetIsLoggedIn = useContext(SetIsLoggedInContext);
  const button = {
    marginRight: "20px",
    fontSize: "1rem",
    fontWeight: "700",
    padding: "0.3rem 1.4rem",
  };
  const handleLogout = () => {
    axios
      .post("http://localhost:3001/logout", { withCredentials: true })
      .then((response) => {
        if (response.status === 200) {
          SetIsLoggedIn(false);
          navigate("/login");
        }
      })
      .catch((error) => {
        console.error("Error logging out:", error);
      });
  };
  return (
    <Button
      variant="contained"
      color="error"
      style={button}
      onClick={handleLogout}
    >
      Logout
    </Button>
  );
};
