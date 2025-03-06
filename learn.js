function toggleSidebar() {
    var sidebar = document.getElementById("sidebar");
    var content = document.getElementById("content");
    var btn = document.querySelector(".openbtn");

    if (sidebar.style.width === "250px") {
        sidebar.style.width = "0";
        content.style.marginLeft = "0";
        btn.style.left = "15px";
    } else {
        sidebar.style.width = "250px";
        content.style.marginLeft = "250px";
        btn.style.left = "265px";
    }
}

document.addEventListener('click', function(event) {
    var sidebar = document.getElementById("sidebar");
    var btn = document.querySelector(".openbtn");

    if (sidebar.style.width === "250px" && !sidebar.contains(event.target) && !btn.contains(event.target)) {
        sidebar.style.width = "0";
        document.getElementById("content").style.marginLeft = "0";
        btn.style.left = "15px";
    }
});
