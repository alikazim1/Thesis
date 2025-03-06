<?php
// Make sure the password is empty if the root user has no password

try{
$con = mysqli_connect("localhost", "root", "", "tutorial") or die("Couldn't connect");}

catch(mysqli_sql_exception){
    echo "Failed to connect to the database";
}

if($con){
    echo "Connected to the database";
}

?>
