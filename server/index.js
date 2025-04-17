const express = require("express");

const mongoose = require("mongoose");

const cors = require("cors");
const session = require("express-session");
const MongooseStore = require("connect-mongo");
const bcrypt = require("bcryptjs");

const UserModel = require("./model/user");
const dotenv = require("dotenv");
const PORT = process.env.PORT || 4000;
dotenv.config();
const app = express();
app.use(express.json());

app.use(
  cors({
    origin: process.env.FRONTEND_URL,
    methods: ["POST", "GET"],
    credentials: true,
  })
);

mongoose
  .connect(process.env.MONGO_URI)
  .then(() => {
    console.log("Connected to MongoDB");
  })
  .catch((err) => {
    console.log("Failed to connect to MongoDB", err);
  });

app.listen(process.env.PORT, () => {
  console.log(`Server is running on port ${process.env.PORT}`);
});

app.use(
  session({
    secret: process.env.SESSION_SECRET,
    resave: false,
    saveUninitialized: true,
    store: MongooseStore.create({
      mongoUrl: process.env.MONGO_URI,
    }),
    cookie: {
      maxAge: 24 * 60 * 60 * 1000, // 1 day
    },
  })
);

app.post("/signup", async (req, res) => {
  try {
    const { name, email, password } = req.body;
    console.log(name + " " + email + " " + password);
    const existingUser = await UserModel.findOne({ email });
    console.log(existingUser);
    if (existingUser) {
      return res.status(400).json({ message: "User already exists" });
    }
    const hashedPassword = await bcrypt.hash(password, 10);
    const newUser = new UserModel({
      name,
      email,
      password: hashedPassword,
    });
    const savedUser = await newUser.save();
    res
      .status(201)
      .json({ message: "User created successfully", user: savedUser });
  } catch (error) {
    console.error("Error during signup:", error);
    res.status(500).json({ message: "Internal server error" });
  }
});

app.post("/login", async (req, res) => {
  try {
    const { email, password } = req.body;
    const user = await UserModel.findOne({ email });
    if (user) {
      const passwordMatch = await bcrypt.compare(password, user.password);
      if (passwordMatch) {
        req.session.user = { id: user._id, name: user.name, email: user.email };
        res.json("Success");
        console.log(email);
      } else {
        res.status(401).json("Password doesn't match");
        // alert("user doesn't exist");
      }
    } else {
      res.status(404).json("No Records found");
    }
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/user", (req, res) => {
  if (req.session.user) {
    res.json({ user: req.session.user });
  } else {
    res.status(401).json("Not authenticated");
  }
});

app.post("/logout", (req, res) => {
  if (req.session) {
    req.session.destroy((err) => {
      if (err) {
        res.status(500).json({ error: "Failed to logout" });
      } else {
        res.status(200).json("Logout successful");
      }
    });
  } else {
    res.status(400).json({ error: "No session found" });
  }
});
