import React from "react";
import { AppBar, Typography, Toolbar, Button } from "@mui/material";
import { Link } from "react-router-dom";
import { Logout } from "./Logout";
import { IsLoggedInContext } from "../App";
import { SetIsLoggedInContext } from "../App";

import "../../src/index.css";

export const Navbar = () => {
  const isLoggedIn = React.useContext(IsLoggedInContext);
  const setIsLoggedIn = React.useContext(SetIsLoggedInContext);

  const button = {
    marginRight: "20px",
    fontSize: "1rem",
    fontWeight: "700",
    padding: "0.3rem 1.4rem",
  };
  return (
    <AppBar sx={{ bgcolor: "#333" }}>
      <Toolbar>
        <Typography variant="h4" component="div" sx={{ flexGrow: 1 }}>
          PyTorch Learning website
        </Typography>
        {!isLoggedIn ? (
          <>
            <Button
              variant="contained"
              style={button}
              color="error"
              component={Link}
              to="/login"
            >
              Login
            </Button>

            <Button
              variant="contained"
              style={button}
              color="success"
              component={Link}
              to="/signup"
            >
              Signup
            </Button>
          </>
        ) : (
          <Logout setIsLoggedIn={setIsLoggedIn} />
        )}
      </Toolbar>
    </AppBar>
  );
};
