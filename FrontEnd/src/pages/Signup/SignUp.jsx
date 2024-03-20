import React from "react";
import "./SignUp.css";
function SignUp() {
  return (
    <>
      <div className="cont">
        <div className="boxItem">
          <h1>Sign Up</h1>
          <div className="name">
            <label>Name:</label>
            <input type="text" placeholder="Enter Name"></input>
          </div>
          <div className="contact name">
            <label>Contact No:</label>
            <input type="number" placeholder="Enter Your Contact"></input>
          </div>
          <div className="email name">
            <label>Email:</label>
            <input type="email" placeholder="Enter Your Email"></input>
          </div>
          <div className="pass name">
            <label>Password:</label>
            <input type="password" placeholder="Enter Password"></input>
            {/* <input type="checkbox" onclick="myFunction()">Show Password</input> */}
          </div>
          <div className="pass name">
            <label>Confirm Password:</label>
            <input type="password" placeholder="Confirm Password"></input>
            {/* <input type="checkbox" onclick="myFunction()">Show Password</input> */}
          </div>
          <div className="btnn">
            <button>Submit</button>
          </div>
          <div className="already">
            <p>Already Registered?</p>
            <a href="/login">Login</a>
          </div>
        </div>
      </div>
    </>
  );
}

export default SignUp;
