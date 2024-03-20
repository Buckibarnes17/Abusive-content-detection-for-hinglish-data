import React from 'react'
import './Login.css'

function login() {

  return (
    <>
      <div className="contain">
        <div className="box">
            <h1>Login</h1>
            <div className="usernam align">
              <label htmlFor="username">Username</label>
              <input type="text" id='username' placeholder='Enter Your Username'/>
            </div>
            <div className="passwor align">
              <label htmlFor="password">Password</label>
              <input type="text" id='password' placeholder='Enter Your Password'/>
            </div>
            <div className="extra">
              <div className="rememb">
                <input type="checkbox" id='checkbox'/>
                <p>Remember me</p>
              </div>
                <a href="/forgotpassword">Forgot Password?</a>
            </div>
            <div className="submi">
              <button type='submit'>Login</button>
            </div>
            <div className="reg">
              Create New Account
              <a href="/signup">SignUp</a>
            </div>
        </div>
      </div>
    </>
  )
}

export default login
