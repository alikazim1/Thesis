// server.js
const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const session = require("express-session");
const MongooseStore = require("connect-mongo");
const dotenv = require("dotenv");
const AuthController = require("./controllers/AuthController");

dotenv.config();

class Server {
  constructor() {
    this.app = express();
    this.port = process.env.PORT || 4000;
    this.connectDB();
    this.middlewares();
    this.routes();
    this.start();
  }

  connectDB() {
    mongoose
      .connect(process.env.MONGO_URI)
      .then(() => console.log("Connected to MongoDB"))
      .catch((err) => console.log("MongoDB connection error", err));
  }

  middlewares() {
    this.app.use(express.json());
    this.app.use(
      cors({
        origin: process.env.FRONTEND_URL,
        methods: ["POST", "GET"],
        credentials: true,
      })
    );

    this.app.use(
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
  }

  routes() {
    const authController = new AuthController();
    this.app.post("/signup", authController.signup);
    this.app.post("/login", authController.login);
    this.app.get("/user", authController.getUser);
    this.app.post("/logout", authController.logout);
  }

  start() {
    this.app.listen(this.port, () => {
      console.log(`Server running on port ${this.port}`);
    });
  }
}

new Server();
