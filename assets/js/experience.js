//  Work experience cards


const experiencecards = document.querySelector(".experience-cards");
const exp = [
 
   {
    title: "Machine learning Engineer, Revca",
    cardImage: "assets/images/experience-page/01.jpg",
    place: "Washington, United States",
    time: "(Feb. 2023 - Present)",
    desp:"<li>Information extraction from scanned receipts</li><li>Vehicle speed detection</li><li>Brain tumor segmentation</li>",
  },
  
  {
    title: "Machine learning Engineer, OpenBee",
    cardImage: "assets/images/experience-page/02.jpg",
    place: "Hammamet, Tunisie",
    time: "(Apr. 2018 - Present)",
    desp: "<li>Multilingual optical character recognition engine OCR</li><li>Barcode detection and recognition</li><li>Document classifier</li><li>Receipt detection</li><li>Sentiment analysis</li> ",
  }
];

const showCards2 = () => {
  let output = "";
  exp.forEach(
    ({ title, cardImage, place, time, desp }) =>
      (output += `        
      <ul>
      <li class="card card1">
        <img src="${cardImage}" class="featured-image"/>
        <article class="card-body">
          <header>
            <div class="title">
              <h3>${title}</h3>
            </div>
            <p class="meta">
              <span class="pre-heading">${place}</span><br>
              <span class="author">${time}</span>
            </p>
            <ol>
              ${desp}
            </ol>
          </header>
        </article>
      </li>
    </ul>
      `)
  );
  experiencecards.innerHTML = output;
};
document.addEventListener("DOMContentLoaded", showCards2);

const showCards = () => {
  let output = "";
  volunteershipcards.forEach(
    ({ title, cardImage, description }) =>
      (output += `        
      <div class="card volunteerCard" style="height: 600px;width:400px">
      
      <img src="${cardImage}" height="300" width="65" class="card-img" style="border-radius:10px">
      <div class="content">
          <h2 class="volunteerTitle">${title}</h2><br>
          <p class="copy">${description}</p></div>
      
      </div>
      `)
  );
  volunteership.innerHTML = output;
};
document.addEventListener("DOMContentLoaded", showCards);


// Mentorship Card


const mentorshipcards = document.querySelector(".mentorship-cards");
const mentor = [
  // {
  //   title: "HakinCode",
  //   image: "assets/images/experience-page/hakin.png",
  //   time: "06/2020 - 08/2020",
  //   desp: "<li>It is an open source community where students and mentors can apply.</li><hr /><li>Ample amount of technologies and projects are there and we are given opportunity to work on them according to our interest and knowledge.</li>",
  // },
  // {
  //   title: "Google Summer of Code",
  //   image: "assets/images/experience-page/gsoc.png",
  //   time: "03/2020 - 08/2020",
  //   desp: "<li>Google Summer of Code is a global program focused on introducing students to open source software development.</li><hr /><li>It is a great platform to explore new areas, maybe discover a new career path!</li>",
  // },
];

const showCards3 = () => {
  let output = "";
  mentor.forEach(
    ({ title, image, time, desp}) =>
      (output += `        
      <div class="column mentorshipCard"> 
      <div class="card card2 mentorshipCardCover">
        <img src="${image}" class="card-img-top" alt="..."  width="64" height="300">
        <div class="information">
        <div class="card-body">
          <h5 class="card-title">${title}</h5>
          <p class=""sub-title">${time}</p>
        </div>
        <div class="more-information">
        <ul class="list-group list-group-flush p-0 right-aligned">
          <div class="list-group-item card2 disclaimer">${desp}</div>
        </ul>
        </div>
        </div>
      </div>
      </div>
      `)
  );
  mentorshipcards.innerHTML = output;
};
document.addEventListener("DOMContentLoaded", showCards3);
