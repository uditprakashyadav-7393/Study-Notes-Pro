<?php
/**
 * chat-proxy.php
 * Upload this to the SAME folder as your site's index.html on InfinityFree.
 * It forwards chat requests to NVIDIA's API, keeping your API key hidden
 * from anyone viewing your site's source code.
 */

// 🔑 Paste your NVIDIA API key here (starts with nvapi-). This file is never
// sent to the browser, so it's safe here.
define('NVIDIA_API_KEY', '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');

// --- CORS: only allow requests from your own site ---
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *'); // tighten to your domain once live, e.g. https://studynotespro.great-site.net
header('Access-Control-Allow-Methods: POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit;
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['error' => ['message' => 'Method not allowed']]);
    exit;
}

// --- Read the JSON body the browser sent (model + messages) ---
$raw = file_get_contents('php://input');
$body = json_decode($raw, true);

if (!$body || !isset($body['messages'])) {
    http_response_code(400);
    echo json_encode(['error' => ['message' => 'Invalid request body']]);
    exit;
}

// --- Basic abuse guard: cap turns and message length ---
if (count($body['messages']) > 20) {
    http_response_code(400);
    echo json_encode(['error' => ['message' => 'Too many messages']]);
    exit;
}
foreach ($body['messages'] as $m) {
    if (isset($m['content']) && strlen($m['content']) > 6000) {
        http_response_code(400);
        echo json_encode(['error' => ['message' => 'Message too long']]);
        exit;
    }
}

// --- Forward to NVIDIA's OpenAI-compatible endpoint ---
$payload = [
    'model'      => $body['model'] ?? 'openai/gpt-oss-120b',
    'messages'   => $body['messages'],
    'max_tokens' => min((int)($body['max_tokens'] ?? 1000), 1000),
];

$ch = curl_init('https://integrate.api.nvidia.com/v1/chat/completions');
curl_setopt_array($ch, [
    CURLOPT_POST           => true,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_HTTPHEADER     => [
        'Content-Type: application/json',
        'Authorization: Bearer ' . NVIDIA_API_KEY,
    ],
    CURLOPT_POSTFIELDS     => json_encode($payload),
    CURLOPT_TIMEOUT        => 60,
]);

$response  = curl_exec($ch);
$httpCode  = curl_getinfo($ch, CURLINFO_HTTP_CODE);
$curlError = curl_error($ch);
curl_close($ch);

if ($response === false) {
    http_response_code(502);
    echo json_encode(['error' => ['message' => 'Upstream request failed: ' . $curlError]]);
    exit;
}

http_response_code($httpCode ?: 500);
echo $response;
