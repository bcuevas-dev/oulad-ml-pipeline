
--
-- Host: localhost    Database: ouladdb
-- ------------------------------------------------------
-- Server version	9.3.0

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `assessments`
--

DROP TABLE IF EXISTS `assessments`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `assessments` (
  `code_module` text,
  `code_presentation` text,
  `id_assessment` bigint DEFAULT NULL,
  `assessment_type` text,
  `date` double DEFAULT NULL,
  `weight` double DEFAULT NULL,
  `assessment_type_ord` bigint DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `courses`
--

DROP TABLE IF EXISTS `courses`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `courses` (
  `code_module` text,
  `code_presentation` text,
  `module_presentation_length` bigint DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `studentAssessment`
--

DROP TABLE IF EXISTS `studentAssessment`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `studentAssessment` (
  `id_assessment` bigint DEFAULT NULL,
  `id_student` bigint DEFAULT NULL,
  `date_submitted` bigint DEFAULT NULL,
  `is_banked` bigint DEFAULT NULL,
  `score` double DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `studentInfo`
--

DROP TABLE IF EXISTS `studentInfo`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `studentInfo` (
  `code_module` text,
  `code_presentation` text,
  `id_student` bigint DEFAULT NULL,
  `gender` text,
  `region` text,
  `highest_education` text,
  `imd_band` text,
  `age_band` text,
  `num_of_prev_attempts` double DEFAULT NULL,
  `studied_credits` double DEFAULT NULL,
  `disability` text,
  `final_result` text,
  `age_band_ord` double DEFAULT NULL,
  `highest_education_ord` bigint DEFAULT NULL,
  `final_result_ord` bigint DEFAULT NULL,
  `gender_ord` bigint DEFAULT NULL,
  `region_ord` bigint DEFAULT NULL,
  `disability_ord` bigint DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `studentRegistration`
--

DROP TABLE IF EXISTS `studentRegistration`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `studentRegistration` (
  `code_module` text,
  `code_presentation` text,
  `id_student` bigint DEFAULT NULL,
  `date_registration` double DEFAULT NULL,
  `date_unregistration` double DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `studentVle`
--

DROP TABLE IF EXISTS `studentVle`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `studentVle` (
  `code_module` text,
  `code_presentation` text,
  `id_student` bigint DEFAULT NULL,
  `id_site` bigint DEFAULT NULL,
  `date` bigint DEFAULT NULL,
  `sum_click` bigint DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `vle`
--

DROP TABLE IF EXISTS `vle`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `vle` (
  `id_site` bigint DEFAULT NULL,
  `code_module` text,
  `code_presentation` text,
  `activity_type` text,
  `week_from` double DEFAULT NULL,
  `week_to` double DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;


-- Dump completed on 2025-07-02  4:15:18
