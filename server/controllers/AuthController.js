// controllers/AuthController.js
const bcrypt = require("bcryptjs");
const UserModel = require("../model/user");

class AuthController {
  async signup(req, res) {
    try {
      const { name, email, password } = req.body;
      const existingUser = await UserModel.findOne({ email });
      if (existingUser) {
        return res.status(400).json({ message: "User already exists" });
      }

      const hashedPassword = await bcrypt.hash(password, 10);
      const newUser = new UserModel({ name, email, password: hashedPassword });
      const savedUser = await newUser.save();
      res.status(201).json({ message: "User created", user: savedUser });
    } catch (error) {
      res.status(500).json({ message: "Signup error" });
    }
  }

  async login(req, res) {
    try {
      const { email, password } = req.body;
      const user = await UserModel.findOne({ email });
      if (!user) return res.status(404).json("No user found");

      const match = await bcrypt.compare(password, user.password);
      if (!match) return res.status(401).json("Incorrect password");

      req.session.user = { id: user._id, name: user.name, email: user.email };
      res.json("Success");
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  getUser(req, res) {
    if (req.session.user) {
      res.json({ user: req.session.user });
    } else {
      res.status(401).json("Not authenticated");
    }
  }

  logout(req, res) {
    if (req.session) {
      req.session.destroy((err) => {
        if (err) return res.status(500).json({ error: "Logout failed" });
        res.status(200).json("Logout successful");
      });
    } else {
      res.status(400).json({ error: "No session" });
    }
  }
}

module.exports = AuthController;
