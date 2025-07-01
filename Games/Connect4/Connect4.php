<!DOCTYPE html>
<?php include_once $_SERVER['DOCUMENT_ROOT'] . '/Include/globals.php';?>
<html lang="en">

<head>
    <?php include_once $GL_root . $GL_path . '/Include/head_includes.php';?>
    <meta name="description" content="" />
    <meta name="keywords" content="" />
    <link rel="canonical" href="https://www.laughingskull.org/Games/Connect4/Connect4.php">
    <title>Connect-4</title>
</head>

<body>
    <?php include_once $GL_root . $GL_path . '/Include/header.php';?>
    <?php include_once $GL_root . $GL_path . '/Include/resolutionAlert.php';?>
    <?php include_once $GL_root . $GL_path . '/Games/Connect4/Connect4.html.php';?>
    <?php include_once $GL_root . $GL_path . '/Include/footer.php';?>

    <!-- JS -->
    <script src='/Code/JS/Library/Engine/Prototype_5_00.js' type="text/javascript"></script>
    <script src='/Code/JS/Library/Engine/ENGINE_5_00.js' type="text/javascript"></script>
    <script src="/Code/JS/Library/Engine/GRID_4_00.js" type="text/javascript"></script>
    <script src="/Code/JS/Library/Engine/GenericTimers_1_03.js" type="text/javascript"></script>
    <script src="/Code/JS/Library/Engine/SUBTITLE_1_00.js" type="text/javascript"></script>
    <script src="/Assets/Definitions/Connect4/assets_connect4.js" type="text/javascript"></script>
    <script src="/Games/Connect4/Connect4_class_extensions.js" type="text/javascript"></script>
    <script src="/Games/Connect4/Connect4.js" type="text/javascript"></script>

    <!-- JS END -->
</body>

</html>