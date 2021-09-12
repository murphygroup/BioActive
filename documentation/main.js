const nav = document.querySelector(".navbar");
const head = document.querySelector(".content-start");
const drops = document.querySelectorAll(".dropdown-content");

const distance = nav.offsetTop;
console.log(distance);

window.onscroll = function() {
	if ( window.pageYOffset > distance ) {
		nav.style.position = "fixed";
		nav.style.top = 0;
		head.style.marginTop = "46px";
		drops.forEach(function(drop) {
			drop.style.position = "fixed";
		})
	} else {
		nav.style.position = "static";
		head.style.marginTop = "0px";
		drops.forEach(function(drop) {
			drop.style.position = "absolute";
		})
	}
}