function searchTags() {
  const query = document.getElementById('searchBar').value;
  fetch(`http://localhost:5000/search?query=${query}`)
  .then(response => response.json())
  .then(data => {
      // Process and display the search results
      console.log(data);
  })
  .catch(error => console.error(`Error: ${error}`));
}
