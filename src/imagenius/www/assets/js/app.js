labels = {
  "N/A": 0,
  "airplane": 5,
  "apple": 53,
  "backpack": 27,
  "banana": 52,
  "baseball bat": 39,
  "baseball glove": 40,
  "bear": 23,
  "bed": 65,
  "bench": 15,
  "bicycle": 2,
  "bird": 16,
  "blender": 83,
  "boat": 9,
  "book": 84,
  "bottle": 44,
  "bowl": 51,
  "broccoli": 56,
  "bus": 6,
  "cake": 61,
  "car": 3,
  "carrot": 57,
  "cat": 17,
  "cell phone": 77,
  "chair": 62,
  "clock": 85,
  "couch": 63,
  "cow": 21,
  "cup": 47,
  "desk": 69,
  "dining table": 67,
  "dog": 18,
  "donut": 60,
  "door": 71,
  "elephant": 22,
  "eye glasses": 30,
  "fire hydrant": 11,
  "fork": 48,
  "frisbee": 34,
  "giraffe": 25,
  "hair drier": 89,
  "handbag": 31,
  "hat": 26,
  "horse": 19,
  "hot dog": 58,
  "keyboard": 76,
  "kite": 38,
  "knife": 49,
  "laptop": 73,
  "microwave": 78,
  "mirror": 66,
  "motorcycle": 4,
  "mouse": 74,
  "orange": 55,
  "oven": 79,
  "parking meter": 14,
  "person": 1,
  "pizza": 59,
  "plate": 45,
  "potted plant": 64,
  "refrigerator": 82,
  "remote": 75,
  "sandwich": 54,
  "scissors": 87,
  "sheep": 20,
  "shoe": 29,
  "sink": 81,
  "skateboard": 41,
  "skis": 35,
  "snowboard": 36,
  "spoon": 50,
  "sports ball": 37,
  "stop sign": 13,
  "street sign": 12,
  "suitcase": 33,
  "surfboard": 42,
  "teddy bear": 88,
  "tennis racket": 43,
  "tie": 32,
  "toaster": 80,
  "toilet": 70,
  "toothbrush": 90,
  "traffic light": 10,
  "train": 7,
  "truck": 8,
  "tv": 72,
  "umbrella": 28,
  "vase": 86,
  "window": 68,
  "wine glass": 46,
  "zebra": 24
}


// Prevent the default form behavior
function handleKeypress(event) {
  if (event.key === "Enter") {
      event.preventDefault();
      searchTags();
  }
}


function searchTags() {
  // Clear previous search results
  document.getElementById('imageResults').innerHTML = '';

  const query = document.getElementById('searchBar').value;
  fetch(`https://127.0.0.1:6000/search?query=${query}`)
  .then(response => response.json())
  .then(data => {
      // Process and display the search results
      Object.keys(data).forEach(imagePath => {
          const imgElement = document.createElement('img');
          const filename = imagePath.split("/").pop();
          imgElement.src = `https://127.0.0.1:6000/image/${filename}`;
          imgElement.alt = 'Found image';
          imgElement.width = 200;  // Set width for display

          // Append to the results container
          document.getElementById('imageResults').appendChild(imgElement);
      });
  })
  .catch(error => console.error(`Error: ${error}`));

  return false; // Prevent the default form behavior
}
