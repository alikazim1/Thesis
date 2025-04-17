import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Grid, Paper, TextField, Typography, Button } from "@mui/material";
import axios from "axios";

export const Signup = () => {
  const heading = {
    fontSize: "2.5rem",
    fontWeight: "bold",
    marginBottom: "20px",
    textAlign: "center",
  };

  const btnStyle = {
    marginTop: "2rem",
    fontSize: "1.1rem",
    fontWeight: "700",
    backgroundColor: "#1976d2",
    borderRadius: "0.5rem",
  };

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate(); // Initialize useNavigate

  const handleSignup = (e) => {
    e.preventDefault(); // Prevent default form submission
    axios
      .post("http://localhost:3001/signup", { name, email, password })
      .then((result) => {
        if (result.status === 201) {
          console.log("User Created Successfully");
          navigate("/login"); // Redirect to login page after successful signup
        }
      })
      .catch((error) => {
        if (error.response && error.response.status === 400) {
          window.alert("User Already Exists");
        } else {
          console.log(error);
          window.alert("Something went wrong, please try again later");
        }
      });
  };

  return (
    <div className="login-wrapper">
      <Grid
        align="center"
        sx={{
          marginLeft: { xs: "0", sm: "5%", md: "10%" }, // Responsive left margin
          transition: "margin 0.3s ease", // Smooth adjustment
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
            backgroundColor: "rgba(255, 255, 255, 0.9)", // Slightly transparent
            padding: 4,
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
            borderRadius: "1rem",
            boxShadow: "10px 10px 20px #ccc",
          }}
        >
          <Typography style={heading}>Signup</Typography>

          <form
            onSubmit={handleSignup} // Prevent default form submission
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "20px",
              width: "100%",
            }}
          >
            <TextField
              onChange={(e) => setName(e.target.value)}
              name="name"
              required
              label="Enter Name"
              type="text"
              fullWidth
              InputLabelProps={{
                style: { fontSize: "1.1rem", fontWeight: 600 },
              }}
            />
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
              Signup
            </Button>
          </form>
        </Paper>
      </Grid>
    </div>
  );
};
