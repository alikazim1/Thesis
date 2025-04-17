import React, { useState, useContext } from "react";
import { Grid, Paper, TextField, Typography, Button } from "@mui/material";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { SetIsLoggedInContext } from "../App";

export const Login = () => {
  const heading = {
    fontSize: "2.5rem",
    fontWeight: "bold",
    marginBottom: "20px",
    textAlign: "center",
  };

  const navigate = useNavigate();
  const setIsLoggedIn = useContext(SetIsLoggedInContext); // Access the context to update login state

  const btnStyle = {
    marginTop: "2rem",
    fontSize: "1.1rem",
    fontWeight: "700",
    backgroundColor: "#1976d2",
    borderRadius: "0.5rem",
  };

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleLogin = (e) => {
    e.preventDefault(); // Prevent the default form submission behavior
    axios
      .post(
        "http://localhost:3001/login",
        { email, password },
        {
          withCredentials: true,
        }
      )
      .then((result) => {
        if (result.data === "Success") {
          axios
            .get("http://localhost:3001/user", { withCredentials: true })
            .then((response) => {
              if (response.data.user) {
                setIsLoggedIn(true); // Update the login state in context
                setEmail(""); // Clear input fields
                setPassword("");
                navigate("/home", { state: { user: response.data.user } });
              }
            });
        }
      })
      .catch((err) => {
        if (err.response) {
          if (err.response.status === 401) {
            alert("Incorrect password.");
          } else if (err.response.status === 404) {
            alert("User doesn't exist.");
          } else {
            alert("Something went wrong.");
          }
        } else {
          alert("Network error.");
        }
      });
  };

  return (
    <div className="login-wrapper">
      <Grid
        align="center"
        sx={{
          marginLeft: { xs: "0", sm: "5%", md: "10%" },
          transition: "margin 0.3s ease",
        }}
      >
        <Paper
          sx={{
            width: {
              xs: "90vw",
              sm: "60vw",
              md: "40vw",
              lg: "30vw",
              xl: "20vw",
            },
            marginRight: "10%",
            backgroundColor: "rgba(255, 255, 255, 0.9)",
            padding: 4,
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
            borderRadius: "1rem",
            boxShadow: "10px 10px 20px #ccc",
          }}
        >
          <Typography style={heading}>Login</Typography>

          <form
            onSubmit={handleLogin}
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "20px",
              width: "100%",
            }}
          >
            <TextField
              onChange={(e) => setEmail(e.target.value)}
              name="email"
              required
              label="Enter Email"
              type="email"
              fullWidth
              InputLabelProps={{
                style: { fontSize: "1.1rem", fontWeight: 600 },
              }}
            />
            <TextField
              onChange={(e) => setPassword(e.target.value)}
              name="password"
              required
              label="Enter Password"
              type="password"
              fullWidth
              InputLabelProps={{
                style: { fontSize: "1.1rem", fontWeight: 600 },
              }}
            />
            <Button type="submit" variant="contained" style={btnStyle}>
              Login
            </Button>
          </form>
        </Paper>
      </Grid>
    </div>
  );
};
