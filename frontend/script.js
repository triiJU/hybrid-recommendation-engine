const API_BASE = "https://hybrid-recommendation-engine.onrender.com"; // change to your Render URL

document.getElementById("fetchBtn").addEventListener("click", async () => {
  const userId = document.getElementById("uid").value;
  const w_cf = document.getElementById("w_cf").value;
  const w_content = document.getElementById("w_content").value;
  const w_neural = document.getElementById("w_neural").value;
  const output = document.getElementById("output");

  if (!userId) {
    output.innerHTML = "<p class='text-red-600'>Please enter a valid user ID.</p>";
    return;
  }

  output.innerHTML = "<p class='text-gray-600'>Loading recommendations...</p>";

  try {
    const res = await fetch(`${API_BASE}/recommend?user_id=${userId}&k=5&w_cf=${w_cf}&w_content=${w_content}&w_neural=${w_neural}`);
    const data = await res.json();
    const recs = data.recommendations || [];

    if (recs.length === 0) {
      output.innerHTML = "<p>No recommendations found for this user.</p>";
      return;
    }

    const list = recs
      .map(r => `<li class='border-b py-1'>${r.title || "Item " + r.item_id} 
                <span class='text-gray-500'>(Score: ${r.score.toFixed(2)})</span></li>`)
      .join("");

    output.innerHTML = `
      <h2 class='text-xl font-semibold mb-2'>Recommendations for User ${userId}</h2>
      <ul class='list-disc ml-5'>${list}</ul>
    `;
  } catch (err) {
    console.error(err);
    output.innerHTML = "<p class='text-red-600'>Failed to fetch recommendations. Check API connection.</p>";
  }
});
