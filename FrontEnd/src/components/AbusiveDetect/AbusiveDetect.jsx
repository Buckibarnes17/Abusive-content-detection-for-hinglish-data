import React from "react";
import "./Abusive.css";
import profile from '../../assets/profile.webp'
function AbusiveDetect() {
  return (
    <>
      <div className="cont">
        <div class="nav">
          <ul class="content">
            <div className="left">
              <li><a href="/">Home</a></li>
              <li><a href="/pricing">Pricing</a></li>
              <div class="dropdown">
                <li>Resources</li>
                <div class="dropdown-options">
                  <a href="#">GetText</a>
                  <a href="#">Blog</a>
                  <a href="#">About Us</a>
                </div>
              </div>
            </div>
            <div className="right">
              <a href='/login'>Login</a>
              <img src={profile} alt="profile icon" />
            </div>
          </ul>
        </div>
        <div class="main">
         <b><h1 class="h">Abusive word detection</h1></b> 
          <textarea class="text" placeholder="Enter the text"></textarea>
        </div>
        <div class="btnn">
          <button class="btn" type="submit">
            Submit
          </button>
        </div>
        <div class="circle"> </div>
        <h3 class="footer">score</h3>
      </div>
    </>
  );
}

export default AbusiveDetect;
